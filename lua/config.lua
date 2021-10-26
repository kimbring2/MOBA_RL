-- Wrapper to read config from the autogenerated file.
dkjson = require( "game/dkjson" )
config_str = require("bots/config_auto")

-- Convert the configuration from string into a table.
local config, pos, err = dkjson.decode(config_str, 1, nil)

if err then
    print("JSON Decode Error=", err " at pos=", pos)
else
    print('Decode sucessful:', config)
end

assert(config.ticks_per_observation ~= nil)
assert(config.game_id ~= nil)
assert(config.hero_picks ~= nil)
assert(#config.hero_picks == 10)

return config