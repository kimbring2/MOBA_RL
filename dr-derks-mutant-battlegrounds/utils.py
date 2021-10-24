import numpy as numpy
import random
import collections
import threading
import time
import timeit
from absl import logging
import tensorflow as tf


MultiHostSettings = collections.namedtuple('MultiHostSettings', 'strategy hosts training_strategy encode decode')
def init_learner_multi_host(num_training_tpus: int):
  """Performs common learner initialization including multi-host setting.

  In multi-host setting, this function will enter a loop for slave learners
  until the master signals end of training.

  Args:
    num_training_tpus: Number of training TPUs.

  Returns:
    A MultiHostSettings object.
  """
  tpu = ''
  job_name = None

  if tf.config.experimental.list_logical_devices('TPU'):
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu, job_name=job_name)
    topology = tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)

    assert num_training_tpus % topology.num_tasks == 0
    num_training_tpus_per_task = num_training_tpus // topology.num_tasks

    hosts = []
    training_coordinates = []
    for per_host_coordinates in topology.device_coordinates:
      host = topology.cpu_device_name_at_coordinates(per_host_coordinates[0], job=job_name)
      task_training_coordinates = (per_host_coordinates[:num_training_tpus_per_task])
      training_coordinates.extend([[c] for c in task_training_coordinates])

      inference_coordinates = per_host_coordinates[num_training_tpus_per_task:]
      hosts.append((host, [topology.tpu_device_name_at_coordinates(c, job=job_name) for c in inference_coordinates]))

    training_da = tf.tpu.experimental.DeviceAssignment(topology, training_coordinates)
    training_strategy = tf.distribute.experimental.TPUStrategy(resolver, device_assignment=training_da)

    return MultiHostSettings(strategy, hosts, training_strategy, tpu_encode, tpu_decode)
  else:
    tf.device('/cpu').__enter__()
    any_gpu = tf.config.experimental.list_logical_devices('GPU')
    device_name = '/device:GPU:0' if any_gpu else '/device:CPU:0'
    strategy = tf.distribute.OneDeviceStrategy(device=device_name)
    enc = lambda x: x
    dec = lambda x, s=None: x if s is None else tf.nest.pack_sequence_as(s, x)

    return MultiHostSettings(strategy, [('/cpu', [device_name])], strategy, enc, dec)


class UnrollStore(tf.Module):
  """Utility module for combining individual environment steps into unrolls."""
  def __init__(self, num_envs, unroll_length, timestep_specs, num_overlapping_steps=0, name='UnrollStore'):
    super(UnrollStore, self).__init__(name=name)
    with self.name_scope:
      self._full_length = num_overlapping_steps + unroll_length + 1

      def create_unroll_variable(spec):
        z = tf.zeros([num_envs, self._full_length] + spec.shape.dims, dtype=spec.dtype)
        return tf.Variable(z, trainable=False, name=spec.name)

      self._unroll_length = unroll_length
      self._num_overlapping_steps = num_overlapping_steps
      self._state = tf.nest.map_structure(create_unroll_variable, timestep_specs)

      # For each environment, the index into the environment dimension of the
      # tensors in self._state where we should add the next element.
      self._index = tf.Variable(tf.fill([num_envs], tf.constant(num_overlapping_steps, tf.int32)), trainable=False, name='index')

  @property
  def unroll_specs(self):
    return tf.nest.map_structure(lambda v: tf.TensorSpec(v.shape[1:], v.dtype), self._state)

  @tf.function
  @tf.Module.with_name_scope
  def append(self, env_ids, values):
    """Appends values and returns completed unrolls.

    Args:
      env_ids: 1D tensor with the list of environment IDs for which we append
        data.
        There must not be duplicates.
      values: Values to add for each environment. This is a structure
        (in the tf.nest sense) of tensors following "timestep_specs", with a
        batch front dimension which must be equal to the length of 'env_ids'.

    Returns:
      A pair of:
        - 1D tensor of the environment IDs of the completed unrolls.
        - Completed unrolls. This is a structure of tensors following
          'timestep_specs', with added front dimensions: [num_completed_unrolls,
          num_overlapping_steps + unroll_length + 1].
    """
    tf.debugging.assert_equal(tf.shape(env_ids), tf.shape(tf.unique(env_ids)[0]), 
                                 message=f'Duplicate environment ids in store {self.name}')
    tf.nest.map_structure(lambda s: tf.debugging.assert_equal(tf.shape(env_ids)[0], tf.shape(s)[0],
                            message=(f'Batch dimension must equal the number of environments in store {self.name}.')), values)
    
    curr_indices = self._index.sparse_read(env_ids)
    unroll_indices = tf.stack([env_ids, curr_indices], axis=-1)
    for s, v in zip(tf.nest.flatten(self._state), tf.nest.flatten(values)):
      s.scatter_nd_update(unroll_indices, v)

    # Intentionally not protecting against out-of-bounds to make it possible to
    # detect completed unrolls.
    self._index.scatter_add(tf.IndexedSlices(1, env_ids))

    return self._complete_unrolls(env_ids)

  @tf.function
  @tf.Module.with_name_scope
  def reset(self, env_ids):
    """Resets state.

    Note, this is only intended to be called when environments need to be reset
    after preemptions. Not at episode boundaries.

    Args:
      env_ids: The environments that need to have their state reset.
    """
    self._index.scatter_update(tf.IndexedSlices(self._num_overlapping_steps, env_ids))

    # The following code is the equivalent of:
    # s[env_ids, :j] = 0
    j = self._num_overlapping_steps
    repeated_env_ids = tf.reshape(tf.tile(tf.expand_dims(tf.cast(env_ids, tf.int64), -1), [1, j]), [-1])

    repeated_range = tf.tile(tf.range(j, dtype=tf.int64), [tf.shape(env_ids)[0]])
    indices = tf.stack([repeated_env_ids, repeated_range], axis=-1)

    for s in tf.nest.flatten(self._state):
      z = tf.zeros(tf.concat([tf.shape(repeated_env_ids), s.shape[2:]], axis=0), s.dtype)
      s.scatter_nd_update(indices, z)

  def _complete_unrolls(self, env_ids):
    # Environment with unrolls that are now complete and should be returned.
    env_indices = self._index.sparse_read(env_ids)
    env_ids = tf.gather(env_ids, tf.where(tf.equal(env_indices, self._full_length))[:, 0])
    env_ids = tf.cast(env_ids, tf.int64)
    unrolls = tf.nest.map_structure(lambda s: s.sparse_read(env_ids), self._state)

    # Store last transitions as the first in the next unroll.
    # The following code is the equivalent of:
    # s[env_ids, :j] = s[env_ids, -j:]
    j = self._num_overlapping_steps + 1
    repeated_start_range = tf.tile(tf.range(j, dtype=tf.int64), [tf.shape(env_ids)[0]])
    repeated_end_range = tf.tile(tf.range(self._full_length - j, self._full_length, dtype=tf.int64), [tf.shape(env_ids)[0]])
    repeated_env_ids = tf.reshape(tf.tile(tf.expand_dims(env_ids, -1), [1, j]), [-1])
    start_indices = tf.stack([repeated_env_ids, repeated_start_range], -1)
    end_indices = tf.stack([repeated_env_ids, repeated_end_range], -1)

    for s in tf.nest.flatten(self._state):
      s.scatter_nd_update(start_indices, s.gather_nd(end_indices))

    self._index.scatter_update(tf.IndexedSlices(1 + self._num_overlapping_steps, env_ids))

    return env_ids, unrolls


class Aggregator(tf.Module):
  """Utility module for keeping state for individual environments."""
  def __init__(self, num_envs, specs, name='Aggregator'):
    """Inits an Aggregator.
    Args:
      num_envs: int, number of environments.
      specs: Structure (as defined by tf.nest) of tf.TensorSpecs that will be
        stored for each environment.
      name: Name of the scope for the operations.
    """
    super(Aggregator, self).__init__(name=name)
    def create_variable(spec):
      z = tf.zeros([num_envs] + spec.shape.dims, dtype=spec.dtype)

      return tf.Variable(z, trainable=False, name=spec.name)

    self._state = tf.nest.map_structure(create_variable, specs)

  @tf.Module.with_name_scope
  def reset(self, env_ids):
    """Fills the tensors for the given environments with zeros."""
    with tf.name_scope('Aggregator_reset'):
      for s in tf.nest.flatten(self._state):
        s.scatter_update(tf.IndexedSlices(0, env_ids))

  @tf.Module.with_name_scope
  def add(self, env_ids, values):
    """In-place adds values to the state associated to the given environments.
    Args:
      env_ids: 1D tensor with the environment IDs we want to add values to.
      values: A structure of tensors following the input spec, with an added
        first dimension that must either have the same size as 'env_ids', or
        should not exist (in which case, the value is broadcasted to all
        environment ids).
    """
    tf.nest.assert_same_structure(values, self._state)
    for s, v in zip(tf.nest.flatten(self._state), tf.nest.flatten(values)):
      s.scatter_add(tf.IndexedSlices(v, env_ids))

  @tf.Module.with_name_scope
  def read(self, env_ids):
    """Reads the values corresponding to a list of environments.
    Args:
      env_ids: 1D tensor with the list of environment IDs we want to read.

    Returns:
      A structure of tensors with the same shapes as the input specs. A
      dimension is added in front of each tensor, with size equal to the number
      of env_ids provided.
    """
    return tf.nest.map_structure(lambda s: s.sparse_read(env_ids), self._state)

  @tf.Module.with_name_scope
  def replace(self, env_ids, values, debug_op_name='', debug_tensors=None):
    """Replaces the state associated to the given environments.
    Args:
      env_ids: 1D tensor with the list of environment IDs.
      values: A structure of tensors following the input spec, with an added
        first dimension that must either have the same size as 'env_ids', or
        should not exist (in which case, the value is broadcasted to all
        environment ids).
      debug_op_name: Debug name for the operation.
      debug_tensors: List of tensors to print when the assert fails.
    """
    tf.debugging.assert_rank(env_ids, 1, message=f'Invalid rank for aggregator {self.name}')

    tf.debugging.Assert(
      tf.reduce_all(tf.equal(tf.shape(env_ids), tf.shape(tf.unique(env_ids)[0]))),
        data=[env_ids, (f'Duplicate environment ids in Aggregator: {self.name} with op name "{debug_op_name}"')] + (debug_tensors or []),
        summarize=4096, name=f'assert_no_dups_{self.name}')

    tf.nest.assert_same_structure(values, self._state)

    for s, v in zip(tf.nest.flatten(self._state), tf.nest.flatten(values)):
      s.scatter_update(tf.IndexedSlices(v, env_ids))


class ProgressLogger(object):
  """Helper class for performing periodic logging of the training progress."""
  def __init__(self,
               summary_writer=None,
               initial_period=0.1,
               period_factor=1.01,
               max_period=10.0,
               starting_step=0):
    """Constructs ProgressLogger.
    Args:
      summary_writer: Tensorflow summary writer to use.
      initial_period: Initial logging period in seconds
        (how often logging happens).
      period_factor: Factor by which logging period is
        multiplied after each iteration (exponential back-off).
      max_period: Maximal logging period in seconds
        (the end of exponential back-off).
      starting_step: Step from which to start the summary writer.
    """
    # summary_writer, last_log_{time, step} are set in reset() function.
    self.summary_writer = None
    self.last_log_time = None
    self.last_log_step = 0
    self.period = initial_period
    self.period_factor = period_factor
    self.max_period = max_period
    # Array of strings with names of values to be logged.
    self.log_keys = []
    self.log_keys_set = set()
    self.step_cnt = tf.Variable(-1, dtype=tf.int64)
    self.ready_values = tf.Variable([-1.0],
                                    dtype=tf.float32,
                                    shape=tf.TensorShape(None))
    self.logger_thread = None
    self.logging_callback = None
    self.terminator = None
    self.reset(summary_writer, starting_step)

  def reset(self, summary_writer=None, starting_step=0):
    """Resets the progress logger.

    Args:
      summary_writer: Tensorflow summary writer to use.
      starting_step: Step from which to start the summary writer.
    """
    self.summary_writer = summary_writer
    self.step_cnt.assign(starting_step)
    self.ready_values.assign([-1.0])
    self.last_log_time = timeit.default_timer()
    self.last_log_step = starting_step

  def start(self, logging_callback=None):
    assert self.logger_thread is None
    self.logging_callback = logging_callback
    self.terminator = threading.Event()
    self.logger_thread = threading.Thread(target=self._logging_loop)
    self.logger_thread.start()

  def shutdown(self):
    assert self.logger_thread
    self.terminator.set()
    self.logger_thread.join()
    self.logger_thread = None

  def log_session(self):
    return []

  def log(self, session, name, value):
    # this is a python op so it happens only when this tf.function is compiled
    if name not in self.log_keys_set:
      self.log_keys.append(name)
      self.log_keys_set.add(name)
    # this is a TF op.
    session.append(value)

  def log_session_from_dict(self, dic):
    session = self.log_session()
    for key in dic:
      self.log(session, key, dic[key])

    return session

  def step_end(self, session, strategy=None, step_increment=1):
    logs = []
    for value in session:
      if strategy:
        value = tf.reduce_mean(tf.cast(strategy.experimental_local_results(value)[0], tf.float32))

      logs.append(value)

    self.ready_values.assign(logs)
    self.step_cnt.assign_add(step_increment)

  def _log(self):
    """Perform single round of logging."""
    logging_time = timeit.default_timer()
    step_cnt = self.step_cnt.read_value()
    if step_cnt == self.last_log_step:
      return

    values = self.ready_values.read_value().numpy()
    if values[0] == -1:
      return

    assert len(values) == len(self.log_keys), 'Mismatch between number of keys and values to log: %r vs %r' % (values, self.log_keys)

    if self.summary_writer:
      self.summary_writer.set_as_default()

    tf.summary.experimental.set_step(step_cnt.numpy())
    if self.logging_callback:
      self.logging_callback()

    for key, value in zip(self.log_keys, values):
      tf.summary.scalar(key, value)

    dt = logging_time - self.last_log_time
    df = tf.cast(step_cnt - self.last_log_step, tf.float32)
    tf.summary.scalar('speed/steps_per_sec', df / dt)
    self.last_log_time, self.last_log_step = logging_time, step_cnt

  def _logging_loop(self):
    """Loop being run in a separate thread."""
    last_log_try = timeit.default_timer()
    while not self.terminator.isSet():
      try:
        self._log()
      except Exception:  
        logging.fatal('Logging failed.', exc_info=True)

      now = timeit.default_timer()
      elapsed = now - last_log_try
      last_log_try = now
      self.period = min(self.period_factor * self.period, self.max_period)
      self.terminator.wait(timeout=max(0, self.period - elapsed))


class StructuredFIFOQueue(tf.queue.FIFOQueue):
  """A tf.queue.FIFOQueue that supports nests and tf.TensorSpec."""
  def __init__(self, capacity, specs, shared_name=None, name='structured_fifo_queue'):
    self._specs = specs
    self._flattened_specs = tf.nest.flatten(specs)
    dtypes = [ts.dtype for ts in self._flattened_specs]
    shapes = [ts.shape for ts in self._flattened_specs]
    super(StructuredFIFOQueue, self).__init__(capacity, dtypes, shapes)

  def dequeue(self, name=None):
    result = super(StructuredFIFOQueue, self).dequeue(name=name)
    return tf.nest.pack_sequence_as(self._specs, result)

  def dequeue_many(self, batch_size, name=None):
    result = super(StructuredFIFOQueue, self).dequeue_many(
        batch_size, name=name)
    return tf.nest.pack_sequence_as(self._specs, result)

  def enqueue(self, vals, name=None):
    tf.nest.assert_same_structure(vals, self._specs)
    return super(StructuredFIFOQueue, self).enqueue(
        tf.nest.flatten(vals), name=name)

  def enqueue_many(self, vals, name=None):
    tf.nest.assert_same_structure(vals, self._specs)
    return super(StructuredFIFOQueue, self).enqueue_many(
        tf.nest.flatten(vals), name=name)


def make_time_major(x):
  """Transposes the batch and time dimensions of a nest of Tensors.

  If an input tensor has rank < 2 it returns the original tensor. Retains as
  much of the static shape information as possible.

  Args:
    x: A nest of Tensors.

  Returns:
    x transposed along the first two dimensions.
  """
  def transpose(t):  
    t_static_shape = t.shape
    if t_static_shape.rank is not None and t_static_shape.rank < 2:
      return t

    t_rank = tf.rank(t)
    t_t = tf.transpose(t, tf.concat(([1, 0], tf.range(2, t_rank)), axis=0))
    t_t.set_shape(tf.TensorShape([t_static_shape[1], t_static_shape[0]]).concatenate(t_static_shape[2:]))
    return t_t

  return tf.nest.map_structure(lambda t: tf.xla.experimental.compile(transpose, [t])[0], x)


def batch_apply(fn, inputs):
  """Folds time into the batch dimension, runs fn() and unfolds the result.

  Args:
    fn: Function that takes as input the n tensors of the tf.nest structure,
      with shape [time*batch, <remaining shape>], and returns a tf.nest
      structure of batched tensors.
    inputs: tf.nest structure of n [time, batch, <remaining shape>] tensors.

  Returns:
    tf.nest structure of [time, batch, <fn output shape>]. Structure is
    determined by the output of fn.
  """
  time_to_batch_fn = lambda t: tf.reshape(t, [-1] + t.shape[2:].as_list())
  batched = tf.nest.map_structure(time_to_batch_fn, inputs)
  output = fn(*batched)
  prefix = [int(tf.nest.flatten(inputs)[0].shape[0]), -1]
  batch_to_time_fn = lambda t: tf.reshape(t, prefix + t.shape[1:].as_list())

  return tf.nest.map_structure(batch_to_time_fn, output)


def get_entity(obs):
    observation_dict = {}

    Hitpoints= obs[0]
    #print("Hitpoints: " + str(Hitpoints))

    Ability0Ready= obs[1]
    #print("Ability0Ready: " + str(Ability0Ready))

    FriendStatueDistance= obs[2]
    FriendStatueAngle= obs[3]
    Friend1Distance= obs[4]
    Friend1Angle= obs[5]
    Friend2Distance= obs[6]
    Friend2Angle= obs[7]
    EnemyStatueDistance= obs[8]
    #print("EnemyStatueDistance: " + str(EnemyStatueDistance))

    EnemyStatueAngle= obs[9]
    Enemy1Distance= obs[10]
    Enemy1Angle= obs[11]
    Enemy2Distance= obs[12]
    Enemy2Angle= obs[13]
    Enemy3Distance= obs[14]
    Enemy3Angle= obs[15]

    HasFocus= obs[16]

    FocusRelativeRotation= obs[17]
    #print("FocusRelativeRotation: " + str(FocusRelativeRotation))
    FocusFacingUs= obs[18]
    #print("FocusFacingUs: " + str(FocusFacingUs))
    FocusFocusingBack= obs[19]
    #print("FocusFocusingBack: " + str(FocusFocusingBack))
    FocusHitpoints= obs[20]
    #print("FocusHitpoints: " + str(FocusHitpoints))

    Ability1Ready= obs[21]
    #print("Ability1Ready: " + str(Ability1Ready))

    Ability2Ready= obs[22]
    #print("Ability2Ready: " + str(Ability2Ready))

    FocusDazed= obs[23]
    FocusCrippled= obs[24]
    HeightFront1= obs[25]
    #print("HeightFront1: " + str(HeightFront1))

    HeightFront5= obs[26]
    #print("HeightFront5: " + str(HeightFront5))

    HeightBack2= obs[27]
    #print("HeightBack2: " + str(HeightBack2))

    PositionLeftRight= obs[28]
    #print("PositionLeftRight: " + str(PositionLeftRight))

    PositionUpDown= obs[29]
    #print("PositionUpDown: " + str(PositionUpDown))

    Stuck= obs[30]
    #print("Stuck: " + str(Stuck))

    UnusedSense31= obs[31]
    HasTalons= obs[32]
    HasBloodClaws= obs[33]
    HasCleavers= obs[34]
    HasCripplers= obs[35]
    HasHealingGland= obs[36]
    HasVampireGland= obs[37]
    HasFrogLegs= obs[38]
    HasPistol= obs[39]
    HasMagnum= obs[40]
    HasBlaster= obs[41]
    HasParalyzingDart= obs[42]
    HasIronBubblegum= obs[43]
    HasHeliumBubblegum= obs[44]
    HasShell= obs[45]
    HasTrombone= obs[46]
    FocusHasTalons= obs[47]
    FocusHasBloodClaws= obs[48]
    FocusHasCleavers= obs[49]
    FocusHasCripplers= obs[50]
    FocusHasHealingGland= obs[51]
    FocusHasVampireGland= obs[52]
    FocusHasFrogLegs= obs[53]
    FocusHasPistol= obs[54]
    FocusHasMagnum= obs[55]
    FocusHasBlaster= obs[56]
    FocusHasParalyzingDart= obs[57]
    FocusHasIronBubblegum= obs[58]
    FocusHasHeliumBubblegum= obs[59]
    FocusHasShell= obs[60]
    FocusHasTrombone= obs[61]
    UnusedExtraSense30= obs[62]
    UnusedExtraSense31= obs[63]
    #print("")

    friend_state_list = []
    friend_state_list.append(FriendStatueDistance)
    friend_state_list.append(FriendStatueAngle)
    friend_state_list.append(Friend1Distance)
    friend_state_list.append(Friend1Angle)
    friend_state_list.append(Friend2Angle)
    friend_state_list.append(Friend2Distance)
    friend_state_list.append(Friend2Angle)

    enermy_state_list = []
    enermy_state_list.append(EnemyStatueDistance)
    enermy_state_list.append(EnemyStatueAngle)
    enermy_state_list.append(Enemy1Distance)
    enermy_state_list.append(Enemy1Angle)
    enermy_state_list.append(Enemy2Distance)
    enermy_state_list.append(Enemy2Angle)
    enermy_state_list.append(Enemy3Distance)
    enermy_state_list.append(Enemy3Angle)

    focus_state_list = []
    focus_state_list.append(HasFocus)
    focus_state_list.append(FocusRelativeRotation)
    focus_state_list.append(FocusFacingUs)
    focus_state_list.append(FocusFocusingBack)
    focus_state_list.append(FocusHitpoints)
    focus_state_list.append(FocusDazed)
    focus_state_list.append(FocusCrippled)

    terrain_state_list = []
    terrain_state_list.append(HeightFront1)
    terrain_state_list.append(HeightFront5)
    terrain_state_list.append(HeightBack2)
    terrain_state_list.append(PositionLeftRight)
    terrain_state_list.append(PositionUpDown)
    terrain_state_list.append(Stuck)

    my_item_state_list = []
    my_item_state_list.append(HasTalons)
    my_item_state_list.append(HasBloodClaws)
    my_item_state_list.append(HasCleavers)
    my_item_state_list.append(HasCripplers)
    my_item_state_list.append(HasHealingGland)
    my_item_state_list.append(HasVampireGland)
    my_item_state_list.append(HasFrogLegs)
    my_item_state_list.append(HasPistol)
    my_item_state_list.append(HasMagnum)
    my_item_state_list.append(HasBlaster)
    my_item_state_list.append(HasParalyzingDart)
    my_item_state_list.append(HasIronBubblegum)
    my_item_state_list.append(HasHeliumBubblegum)
    my_item_state_list.append(HasShell)
    my_item_state_list.append(HasTrombone)

    focus_item_state_list = []
    focus_item_state_list.append(FocusHasTalons)
    focus_item_state_list.append(FocusHasBloodClaws)
    focus_item_state_list.append(FocusHasCleavers)
    focus_item_state_list.append(FocusHasCripplers)
    focus_item_state_list.append(FocusHasHealingGland)
    focus_item_state_list.append(FocusHasVampireGland)
    focus_item_state_list.append(FocusHasFrogLegs)
    focus_item_state_list.append(FocusHasPistol)
    focus_item_state_list.append(FocusHasMagnum)
    focus_item_state_list.append(FocusHasBlaster)
    focus_item_state_list.append(FocusHasParalyzingDart)
    focus_item_state_list.append(FocusHasIronBubblegum)
    focus_item_state_list.append(FocusHasHeliumBubblegum)
    focus_item_state_list.append(FocusHasShell)
    focus_item_state_list.append(FocusHasTrombone)

    print("##################################################################")
    print("Hitpoints: " + str(Hitpoints))
    print("---------------------------------------------------------------")
    print("Ability0Ready: " + str(Ability0Ready))
    print("Ability1Ready: " + str(Ability1Ready))
    print("Ability2Ready: " + str(Ability2Ready))
    print("---------------------------------------------------------------")
    print("FriendStatueDistance: " + str(FriendStatueDistance))
    print("FriendStatueAngle: " + str(FriendStatueAngle))
    print("Friend1Distance: " + str(Friend1Distance))
    print("Friend1Angle: " + str(Friend1Angle))
    print("Friend2Angle: " + str(Friend2Angle))
    print("Friend2Distance: " + str(Friend2Distance))
    print("Friend2Angle: " + str(Friend2Angle))
    print("---------------------------------------------------------------")
    print("EnemyStatueDistance: " + str(EnemyStatueDistance))
    print("EnemyStatueAngle: " + str(EnemyStatueAngle))
    print("Enemy1Distance: " + str(Enemy1Distance))
    print("Enemy1Angle: " + str(Enemy1Angle))
    print("Enemy2Distance: " + str(Enemy2Distance))
    print("Enemy2Angle: " + str(Enemy2Angle))
    print("Enemy3Distance: " + str(Enemy3Distance))
    print("Enemy3Angle: " + str(Enemy3Angle))
    print("---------------------------------------------------------------")
    print("HasFocus: " + str(HasFocus))
    print("FocusRelativeRotation: " + str(FocusRelativeRotation))
    print("FocusFacingUs: " + str(FocusFacingUs))
    print("FocusFocusingBack: " + str(FocusFocusingBack))
    print("FocusHitpoints: " + str(FocusHitpoints))
    print("FocusDazed: " + str(FocusDazed))
    print("FocusCrippled: " + str(FocusCrippled))
    print("---------------------------------------------------------------")
    print("HeightFront1: " + str(HeightFront1))
    print("HeightFront5: " + str(HeightFront5))
    print("HeightBack2: " + str(HeightBack2))
    print("PositionLeftRight: " + str(PositionLeftRight))
    print("PositionUpDown: " + str(PositionUpDown))
    print("Stuck: " + str(Stuck))
    print("---------------------------------------------------------------")
    print("HasTalons: " + str(HasTalons))
    print("HasBloodClaws: " + str(HasBloodClaws))
    print("HasCleavers: " + str(HasCleavers))
    print("HasCripplers: " + str(HasCripplers))
    print("HasHealingGland: " + str(HasHealingGland))
    print("HasVampireGland: " + str(HasVampireGland))
    print("HasFrogLegs: " + str(HasFrogLegs))
    print("HasPistol: " + str(HasPistol))
    print("HasMagnum: " + str(HasMagnum))
    print("HasBlaster: " + str(HasBlaster))
    print("HasParalyzingDart: " + str(HasParalyzingDart))
    print("HasIronBubblegum: " + str(HasIronBubblegum))
    print("HasHeliumBubblegum: " + str(HasHeliumBubblegum))
    print("HasShell: " + str(HasShell))
    print("HasTrombone: " + str(HasTrombone))
    print("---------------------------------------------------------------")
    print("FocusHasTalons: " + str(FocusHasTalons))
    print("FocusHasBloodClaws: " + str(FocusHasBloodClaws))
    print("FocusHasCleavers: " + str(FocusHasCleavers))
    print("FocusHasCripplers: " + str(FocusHasCripplers))
    print("FocusHasHealingGland: " + str(FocusHasHealingGland))
    print("FocusHasVampireGland: " + str(FocusHasVampireGland))
    print("FocusHasFrogLegs: " + str(FocusHasFrogLegs))
    print("FocusHasPistol: " + str(FocusHasPistol))
    print("FocusHasMagnum: " + str(FocusHasMagnum))
    print("FocusHasBlaster: " + str(FocusHasBlaster))
    print("FocusHasParalyzingDart: " + str(FocusHasParalyzingDart))
    print("FocusHasIronBubblegum: " + str(FocusHasIronBubblegum))
    print("FocusHasHeliumBubblegum: " + str(FocusHasHeliumBubblegum))
    print("FocusHasShell: " + str(FocusHasShell))
    print("FocusHasTrombone: " + str(FocusHasTrombone))
    print("---------------------------------------------------------------")
    print("UnusedSense31: " + str(UnusedSense31))
    print("UnusedExtraSense30: " + str(UnusedExtraSense30))
    print("UnusedExtraSense31: " + str(UnusedExtraSense31))
    print("##################################################################")
    print("")
    
    #return entity


def get_action(entity=None):
    # These are the actions a Derkling can take, which you send to the step function.

    # A number between -1 and 1
    MoveX = random.uniform(-1.0, 1.0)

    # A number between -1 and 1
    Rotate = random.uniform(-1.0, 1.0)

    # A number between 0 and 1
    ChaseFocus = random.uniform(0.0, 1.0)

    # 0=don’t cast. 1-3=cast corresponding ability
    CastingSlot = random.randint(1, 2)

    # 0=keep current focus. 1=focus home statue. 2-3=focus teammates, 4=focus enemy statue, 5-7=focus enemy
    ChangeFocus = random.randint(4, 7)

    return (MoveX, Rotate, ChaseFocus, CastingSlot, ChangeFocus)

