
local PurchaseItem = {}

PurchaseItem.Name = "Purchase Item"
PurchaseItem.NumArgs = 3


local tableItemsToBuy = { 
				"item_tango",
				"item_tango",
			};


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

----------------------------------------------------------------------------------------------------

function PurchaseItem:Call( hUnit, item, item_name )
    --hUnit.sideShopMode = true;
    --print("type(item): ", type(item))
    --print("item: ", item)
    --print("dump(item): ", dump(item))
    --print("tostring(item_name): ", tostring(item_name))

	if ( #tableItemsToBuy == 0 )
	then
		hUnit:SetNextItemPurchaseValue( 0 );
		return;
	end

	local sNextItem = tableItemsToBuy[1];
	--local sNextItem = "item_tango"

	hUnit:SetNextItemPurchaseValue( GetItemCost( sNextItem ) );

	if ( hUnit:GetGold() >= GetItemCost( sNextItem ) )
	then
		hUnit:ActionImmediate_PurchaseItem( sNextItem );
		table.remove( tableItemsToBuy, 1 );
	end

end

----------------------------------------------------------------------------------------------------
return PurchaseItem

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
