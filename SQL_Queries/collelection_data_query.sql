/* SQL used to create collection_level_data.csv */
/* Database from Flipside Crypto */

with mints as (
  select 
  token_metadata:collection_name as collection_name,
  token_metadata:artist as artist,
  token_metadata:"aspect ratio (w/h)" as aspect_ratio,
  token_metadata:curation_status as curation_status,
  token_metadata:"is dynamic" as is_dynamic,
  token_metadata:"script type" as script_type,
  token_metadata:series as series,
  token_metadata:"uses hash" as use_hash,
  count(ene.token_id) as count_token,
  round(avg(datediff(day, created_at_timestamp, '2021-09-01T00:00:00Z'))) as days_since_mint
  --count(token_metadata:traits) as trait_count,
  from  ethereum.nft_events as ene
  inner join ethereum.nft_metadata as mt on ene.contract_address = mt.contract_address and ene.token_id = mt.token_id
  where  ene.block_timestamp is not null
      and ene.contract_address in ('0x059edd72cd353df5106d2b9cc5ab83a52287ac3a','0xa7d8d9ef8d8ce8992df33d8b8cf4aebabd5bd270')
        and event_type = 'mint'
      --and mt.token_metadata:collection_name::string = 'Chromie Squiggle by Snowfro'  
group by 1,2,3,4,5,6,7,8
order by 1
)
,
d_feature AS(
  select distinct
  token_metadata:collection_name as collection_name,
  token_metadata:features as feature
  from  ethereum.nft_events as ene
  inner join ethereum.nft_metadata as mt on ene.contract_address = mt.contract_address and ene.token_id = mt.token_id
  where  ene.block_timestamp is not null
      and ene.contract_address in ('0x059edd72cd353df5106d2b9cc5ab83a52287ac3a','0xa7d8d9ef8d8ce8992df33d8b8cf4aebabd5bd270')
        and event_type = 'mint'
)
,
n_feature as(
  select 
    collection_name,
    (len(feature) - len(replace(feature,',','')))+1 as feature_number
  from d_feature
  order by collection_name
),
feature as (
  select collection_name, max(feature_number) as feature_number
  from n_feature
  group by collection_name
  order by collection_name
),
d_trait AS(
  select distinct
  token_metadata:collection_name as collection_name,
  token_metadata:traits as traits
  from  ethereum.nft_events as ene
  inner join ethereum.nft_metadata as mt on ene.contract_address = mt.contract_address and ene.token_id = mt.token_id
  where  ene.block_timestamp is not null
      and ene.contract_address in ('0x059edd72cd353df5106d2b9cc5ab83a52287ac3a','0xa7d8d9ef8d8ce8992df33d8b8cf4aebabd5bd270')
        and event_type = 'mint'
),
n_trait as(
  select 
    collection_name,
    (len(traits) - len(replace(traits,',','')))+1 as traits_number
  from d_trait
  order by collection_name
), 
trait as(
  select collection_name, max(traits_number) as traits_number
  from n_trait
  group by collection_name
  order by collection_name
),
  mint_info as(
  select 
  token_metadata:collection_name as collection_name
    , tx_currency as mint_currency
    , round(datediff(day, min(block_timestamp), max(block_timestamp))+1) as mint_duration
  from  ethereum.nft_events as ene
  inner join ethereum.nft_metadata as mt on ene.contract_address = mt.contract_address and ene.token_id = mt.token_id
  where  ene.block_timestamp is not null
  and ene.contract_address in ('0x059edd72cd353df5106d2b9cc5ab83a52287ac3a','0xa7d8d9ef8d8ce8992df33d8b8cf4aebabd5bd270')
  and event_type = 'mint'
  group by 1,2
),
august_sale as (
  select 
  token_metadata:collection_name as collection_name
    , count(mt.token_id) as August_sale_count
    , avg(price_usd) as August_sale_price
  from  ethereum.nft_events as ene
  inner join ethereum.nft_metadata as mt on ene.contract_address = mt.contract_address and ene.token_id = mt.token_id
  where  ene.block_timestamp is not null
      and ene.contract_address in ('0x059edd72cd353df5106d2b9cc5ab83a52287ac3a','0xa7d8d9ef8d8ce8992df33d8b8cf4aebabd5bd270')
        and event_type = 'sale'
    and year(block_timestamp) || '-' || month(block_timestamp) = '2021-8'
group by 1
order by 1
)
/*
price as(
  select
    token_metadata:collection_name as collection_name
    , year(block_timestamp) || '-' || month(block_timestamp) as year_month
    , avg(price_usd) as price_usd
    , max(price_usd) - min(price_usd) as price_range
  from  ethereum.nft_events as ene
  inner join ethereum.nft_metadata as mt on ene.contract_address = mt.contract_address and ene.token_id = mt.token_id
  where  ene.block_timestamp is not null
      and ene.contract_address in ('0x059edd72cd353df5106d2b9cc5ab83a52287ac3a','0xa7d8d9ef8d8ce8992df33d8b8cf4aebabd5bd270')
        and event_type = 'sale'
  group by 1, 2
),
price_pvt as(
  select collection_name,
  sum(case when year_month = '2020-12' then price_usd else 0 end) as Dec20,
  sum(case when year_month = '2021-1' then price_usd else 0 end) as Jan21,
  sum(case when year_month = '2021-2' then price_usd else 0 end) as Feb21,
  sum(case when year_month = '2021-3' then price_usd else 0 end) as Mar21,
  sum(case when year_month = '2021-4' then price_usd else 0 end) as Apr21,
  sum(case when year_month = '2021-5' then price_usd else 0 end) as May21,
  sum(case when year_month = '2021-6' then price_usd else 0 end) as Jun21,
  sum(case when year_month = '2021-7' then price_usd else 0 end) as Jul21,
  sum(case when year_month = '2021-8' then price_usd else 0 end) as Aug21,
  sum(case when year_month = '2021-9' then price_usd else 0 end) as Sep21
  from price
  group by 1
  order by 1
)
*/
select mints.*
  , feature.feature_number
  , trait.traits_number
  , mint_info.mint_currency
  , mint_info.mint_duration
  , august_sale.August_sale_count
  , august_sale.August_sale_price
  /*
  , price_pvt.Dec20
  , price_pvt.Jan21
  , price_pvt.Feb21
  , price_pvt.Mar21
  , price_pvt.Apr21
  , price_pvt.May21
  , price_pvt.Jun21
  , price_pvt.Jul21
  , price_pvt.Aug21
  , price_pvt.Sep21
*/
from mints 
left join feature on mints.collection_name = feature.collection_name
left join trait on mints.collection_name = trait.collection_name
left join mint_info on mints.collection_name = mint_info.collection_name 
left join august_sale on mints.collection_name = august_sale.collection_name 
--left join price_pvt on mints.collection_name = price_pvt.collection_name 

