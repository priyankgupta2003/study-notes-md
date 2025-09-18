
Cumulative table design,

**Full_outer_join** (Today, Yesterday) -> Coalesce the ids and unchanging dimension, compute cumulation (e.g. days since x), combine arrays and changing values. -> Cumulated Output.

**Strengths**
 -Historical analysis without shuffle
 -easy transition analysis

**Drawbacks**
 -can only be backfilled sequentially
 -Handling PII data can be a mess since deleted/inactive users get carried forward



**The compactness vs usability tradeoff**

 The most usable tables usually
  - Have no complex data types

 The most compact tables (not human readable)
  - Are compressed to be as small as possible and can't be queried directly until they're decoded

 The middle-ground tables
  - Use complex data types (e.g. Array, Map, Struct), making querying trickier but also compacting more

When to use the each type of tables
 - Most compact
	- Online systems where latency and data volumes matter a lot. Comsumers are highly technical

- Middle ground
	- Upstream staging/ master data where the majority of consumers are other data engineers

- Most Usable
	- When analytics is the main consumer and the majority of consumers are less technical

**Struct vs Array vs Map**

- Struct 
	- Keys are rigid defined, compression is good
	- Values can be of any type
- Map
	- Keys are loosely defined, compression is okay
	- values all have to be same type
- Array
	- Ordinal
	- List of values that all have to be same data type


**Temporal Cardinality of Explosions of Dimensions**
 - How you can manage when the dimension has a time component to it
 - e.g.: Airbnb has listing, and those listings has calendar and then that calendar hqas a bunch of nights and then you have what it's called a listing night which is its own dimension in some ways, it's like its own entity the night


	- Airbnb has ~6 million listing
		- if we want to know the nightly pricing and availability of each night for the next year that would be 365 * 6 million ~= 2 billion nights
	- Should this dataset be:
		- Listing-level with an array of nights ?
		- Listing night level with 2 billion rows ?
	- so do you keep everything at the listing level or 
	- do you Compact and then you have like listing ID and then an array of night or 
	- do you uh explode it out and then you have two billion rows where you have listing ID and night on the same row

- If you do the sorting right, and use Parquet, will keep this two about same size.

trade-offs here that are interesting and one of the things that can happen here
is like if you use Parquet the right way and everything is sorted and these two data sets are actually the exact same size because of the fact that uh the listing ID will like uh if there's 365 nights but they're all sorted in in in an order then the listing ID will be compressed down because you'll have 365 IDs in a row and that's what run length
encoding is -  all about when you have duplicates of the same value in a column they can be smashed up together as like one entity and then the rest of them can be nullified and just the first value is left -  it'll be like okay we have the same ID 365 times and then the all the rest of them can be removed from the data set and that's one of the most powerful ways to compress data with parquet


- **Drawbacks of denormalized temporal dimensions**
	- If you explode it out and need to join other dimensions, spark shuffle will ruin the compression


- Run-Length Encoding compression
	- 





