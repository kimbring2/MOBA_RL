-------------------------------------------------------------------------------
--- AUTHOR: Nostrademous
-------------------------------------------------------------------------------

local Chat = {}

Chat.Name = "Chat"
Chat.NumArgs = 3

function dump(o)
   if type(o) == 'table' then
      local s = '{ '
      for k,v in pairs(o) do
         if type(k) ~= 'number' then k = '"'..k..'"' end
         s = s .. '['..k..'] = ' .. dump(v) .. ','
      end
      return s .. '} '
   else
      return tostring(o)
   end
end

-------------------------------------------------

function Chat:Call( hHero, sMsg, bAllChat )
    --print("dump(sMsg): ", dump(sMsg))
    --print("dump(bAllChat): ", dump(bAllChat))

    --hHero:ActionImmediate_Chat(sMsg[1], bAllChat[1])
    hHero:ActionImmediate_Chat(sMsg[1], true)
end

-------------------------------------------------

return Chat
