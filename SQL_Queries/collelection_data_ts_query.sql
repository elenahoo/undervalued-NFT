/* SQL Code - Collection Level Data Time-series (can't run here) */
/* Database from Flipside Crypto */

select
    token_metadata:collection_name as collection_name
    , year(block_timestamp) || '-' || month(block_timestamp) as year_month
    , count(ene.token_id) as sale_count
    , avg(price_usd) as price_usd
    , max(price_usd) - min(price_usd) as price_range
  from  ethereum.nft_events as ene
  inner join ethereum.nft_metadata as mt on ene.contract_address = mt.contract_address and ene.token_id = mt.token_id
  where  ene.block_timestamp is not null
    	and ene.contract_address in ('0x059edd72cd353df5106d2b9cc5ab83a52287ac3a','0xa7d8d9ef8d8ce8992df33d8b8cf4aebabd5bd270')
        and event_type = 'sale'
  group by 1, 2
  order by 1, 2
