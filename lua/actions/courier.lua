-------------------------------------------------------------------------------
--- AUTHOR: Nostrademous
-------------------------------------------------------------------------------

local ActionCourier = {}

ActionCourier.Name = "Courier Action"
ActionCourier.NumArgs = 3

-------------------------------------------------


-- KNOWN COURIER ACTIONS ENUMS
--
-- COURIER_ACTION_RETURN == 0               -- RETURN TO FOUNTAIN
-- COURIER_ACTION_SECRET_SHOP == 1          -- GO TO YOUR SECRET SHOP
-- COURIER_ACTION_RETURN_STASH_ITEMS == 2   -- RETURN ITEMS ON COURIER TO HERO'S STASH
-- COURIER_ACTION_TAKE_STASH_ITEMS == 3     -- TAKE HERO'S STASH ITEMS ONTO COURIER
-- COURIER_ACTION_TRANSFER_ITEMS == 4       -- MOVE ITEMS FROM COURIER TO HERO
-- COURIER_ACTION_BURST == 5                -- PROC INVULN SHIELD
-- COURIER_TAKE_AND_TRANSFER_ITEMS == 6     -- FLY TO HERO AND TRANSFER ITEMS ON COURIER TO HERO
-- COURIER_ACTION_ENEMY_SECRET_SHOP == 7    -- GO TO ENEMY'S SECRET SHOP
-- COURIER_ACTION_SIDESHOP == 8             -- GO TO RADIANT SIDE SHOP
-- COURIER_ACTION_SIDESHOP_2 == 9           -- GO TO DIRE SIDE SHOP


-- NOTE: To move the courier you use a hUnit:MoveToLocation() as per normal unit
--       using the handle to the courier. This function is for Courier specific API

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



function ActionCourier:Call(hHero, iCourierAction)
    print("dump(iCourierAction): ", dump(iCourierAction))
    print("dump(GetNumCouriers()): ", dump(GetNumCouriers()))

    if GetNumCouriers() == 0 then return end
    if hHero:IsIllusion() then return end
    local courier = nil
    for i = 0, GetNumCouriers() do
        local t = GetCourier(i)
        print("dump(hHero:GetPlayerID()): ", dump(hHero:GetPlayerID()))
        if hHero:GetPlayerID() == t:GetPlayerID() then
        	print("hHero:GetPlayerID() == t:GetPlayerID()")
            courier = t
            break
        end
    end

    if courier == nil then return end

    --print("dump(courier): ", dump(courier))
    local state = GetCourierState(courier)
    --print("dump(state): ", dump(state))
    --print("dump(COURIER_STATE_DEAD): ", dump(COURIER_STATE_DEAD))
    --print("dump(courier:GetHealth()): ", dump(courier:GetHealth()))

    if state == COURIER_STATE_DEAD or courier:GetHealth() < 1 then
        return
    else
        hHero:ActionImmediate_Courier(courier, iCourierAction[1])
    end
end


-------------------------------------------------

return ActionCourier
