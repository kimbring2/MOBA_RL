-------------------------------------------------------------------------------
--- AUTHOR: Nostrademous
-------------------------------------------------------------------------------

local ToggleAbility = {}

ToggleAbility.Name = "Toggle Ability"
ToggleAbility.NumArgs = 2

-------------------------------------------------
function ToggleAbility:Call( hUnit, intAbilitySlot )
    --local hAbility = hUnit:GetAbilityInSlot(intAbilitySlot[1])

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
    hAbility:ToggleAbility()
end
-------------------------------------------------

return ToggleAbility
