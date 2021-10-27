-------------------------------------------------------------------------------
--- AUTHOR: Nostrademous
-------------------------------------------------------------------------------

local X = {}
local UseAbilityOnTree = {}

UseAbilityOnTree.Name = "Use Ability On Tree"
UseAbilityOnTree.NumArgs = 4


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
function UseAbilityOnTree:Call( hUnit, intAbilitySlot, intTree, iType )
    --print("dump(intAbilitySlot): ", dump(intAbilitySlot))
    --print("dump(intAbilitySlot[1]): ", dump(intAbilitySlot[1]))
    --hItem = hUnit:HasItemInInventory("item_tango")
    local hItem_1 = hUnit:GetItemInSlot(0)
    local hItem_2 = hUnit:GetItemInSlot(1)
    --local hAbility_3 = hUnit:GetAbilityInSlot(9)
    --print("dump(hItem_1): ", dump(hItem_1))
    --print("dump(hItem_2): ", dump(hItem_2))

    local hAbility = hUnit:GetAbilityInSlot(intAbilitySlot[1])
    if not hAbility then
        print('[ERROR]: ', hUnit:GetUnitName(), " failed to find ability in slot ", intAbilitySlot[1])
        do return end
    end

    intTree = intTree[1]
    iType = iType[1]

    -- Note: we do not test if the tree can be ability-targeted due to
    -- range, mana/cooldowns or any debuffs on the hUnit (e.g., silenced).
    -- We assume only valid and legal actions are agent selected

    local vLoc = GetTreeLocation(intTree)
    --print("dump(vLoc): ", dump(vLoc))

    DebugDrawCircle(vLoc, 25, 255, 0, 0)
    DebugDrawLine(hUnit:GetLocation(), vLoc, 255, 0, 0)

    hAbility = hItem_1
    if iType == nil or iType == ABILITY_STANDARD then
        hUnit:Action_UseAbilityOnTree(hAbility, intTree)
    elseif iType == ABILITY_PUSH then
        hUnit:ActionPush_UseAbilityOnTree(hAbility, intTree)
    elseif iType == ABILITY_QUEUE then
        hUnit:ActionQueue_UseAbilityOnTree(hAbility, intTree)
    end
end
-------------------------------------------------

return UseAbilityOnTree
