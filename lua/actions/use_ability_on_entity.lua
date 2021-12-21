-------------------------------------------------------------------------------
--- AUTHOR: Nostrademous
-------------------------------------------------------------------------------

local UseAbilityOnEntity = {}

UseAbilityOnEntity.Name = "Use Ability On Entity"
UseAbilityOnEntity.NumArgs = 4


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
function UseAbilityOnEntity:Call( hUnit, intAbilitySlot, hTarget, iType )
    hTarget = hTarget[1]
    if hTarget == -1 then -- Invalid target. Do nothing.
        do return end
    end
    hTarget = GetBotByHandle(hTarget)

    local hAbility

    if intAbilitySlot[1] >= 0 then
        hAbility = hUnit:GetAbilityInSlot(intAbilitySlot[1])
    else
        itemSlot = -intAbilitySlot[1] - 1
        --print("dump(itemSlot): ", dump(itemSlot))
        hAbility = hUnit:GetItemInSlot(itemSlot)
    end
    if not hAbility then
        print('[ERROR]: ', hUnit:GetUnitName(), " failed to find ability in slot ", intAbilitySlot[1])
        do return end
    end

    iType = iType[1]
 
    -- Note: we do not test if target can be ability-targeted due 
    -- to modifiers (e.g., spell-immunity), range, mana/cooldowns
    -- or any debuffs on the hUnit (e.g., silenced). We assume
    -- only valid and legal actions are agent selected
    if not hTarget:IsNull() and hTarget:IsAlive() then
        local vLoc = hTarget:GetLocation()
        DebugDrawCircle(vLoc, 25, 255, 0, 0)
        DebugDrawLine(hUnit:GetLocation(), vLoc, 255, 0, 0)
        
        if iType == nil or iType == ABILITY_STANDARD then
            hUnit:Action_UseAbilityOnEntity(hAbility, hTarget)
        elseif iType == ABILITY_PUSH then
            hUnit:ActionPush_UseAbilityOnEntity(hAbility, hTarget)
        elseif iType == ABILITY_QUEUE then
            hUnit:ActionQueue_UseAbilityOnEntity(hAbility, hTarget)
        end
    end
end
-------------------------------------------------

return UseAbilityOnEntity
