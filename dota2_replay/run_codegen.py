from grpc_tools import protoc

protoc.main((
    '',
    '-Iproto',
    '--python_out=python_out',
    '--grpc_python_out=python_out',
    'proto/dota_shared_enums.proto',
))