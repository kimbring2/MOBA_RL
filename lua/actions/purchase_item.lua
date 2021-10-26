
local PurchaseItem = {}

PurchaseItem.Name = "Purchase Item"
PurchaseItem.NumArgs = 3


local tableItemsToBuy = { 
				"item_tango",
				"item_tango",
				"item_clarity",
				"item_clarity",
				"item_branches",
				"item_branches",
				"item_magic_stick",
				"item_circlet",
				"item_boots",
				"item_energy_booster",
				"item_staff_of_wizardry",
				"item_ring_of_regen",
				"item_recipe_force_staff",
				"item_point_booster",
				"item_staff_of_wizardry",
				"item_ogre_axe",
				"item_blade_of_alacrity",
				"item_mystic_staff",
				"item_ultimate_orb",
				"item_void_stone",
				"item_staff_of_wizardry",
				"item_wind_lace",
				"item_void_stone",
				"item_recipe_cyclone",
				"item_cyclone",
			};


----------------------------------------------------------------------------------------------------

function PurchaseItem:Call(hUnit, item, item_name)
	--local npcBot = GetBot();
    hUnit.sideShopMode = true;

	if ( #tableItemsToBuy == 0 )
	then
		hUnit:SetNextItemPurchaseValue( 0 );
		return;
	end

	local sNextItem = tableItemsToBuy[1];

	hUnit:SetNextItemPurchaseValue( GetItemCost( sNextItem ) );

	if ( hUnit:GetGold() >= GetItemCost( sNextItem ) )
	then
		hUnit:ActionImmediate_PurchaseItem( sNextItem );
		table.remove( tableItemsToBuy, 1 );
	end

end

----------------------------------------------------------------------------------------------------
return PurchaseItem
