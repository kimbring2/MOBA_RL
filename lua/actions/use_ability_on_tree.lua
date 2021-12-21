-------------------------------------------------------------------------------
--- AUTHOR: Nostrademous
-------------------------------------------------------------------------------

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
    --hItem = hUnit:HasItemInInventory("item_tango")
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

    intTree = intTree[1]
    iType = iType[1]

    -- Note: we do not test if the tree can be ability-targeted due to
    -- range, mana/cooldowns or any debuffs on the hUnit (e.g., silenced).
    -- We assume only valid and legal actions are agent selected
    local tableNearbyTrees = hUnit:GetNearbyTrees(500);
    --print("dump(tableNearbyTrees): ", dump(tableNearbyTrees))
    if tableNearbyTrees[1] then
        intTree = tableNearbyTrees[1]
    end

    --print("dump(intTree): ", dump(intTrees))

    local vLoc = GetTreeLocation(intTree)

    DebugDrawCircle(vLoc, 25, 255, 0, 0)
    DebugDrawLine(hUnit:GetLocation(), vLoc, 255, 0, 0)

    --hAbility = hItem_1
    --print("dump(hAbility): ", dump(hAbility))
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
