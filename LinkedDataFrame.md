# Linked DataFrames

`cheval.LinkedDataFrame` is a subclass of `pandas.DataFrame`, which supports linking frames to other frames. These links can be accessed like normal columns or attributes of the LinkedDataFrame, even for many-to-one and one-to-many relationships. For example, a common practice in agent-based modelling is to have a table of Households and a table of Persons. Using LinkedDataFrame, persons can access attributes of the household (for example, income) using a simple syntax: `persons.household.income`.

Let's say you want to represent a population of Vehicles and Households, where a Vehicle belongs to a Household, and a Household has a collection of Vehicles. Your data looks like this:

`vehicles: LinkedDataFrame`

index|household_id|vehicle_id|manufacturer|model_year|km_travelled
-----|------------|----------|------------|----------|------------
0|0|0|Honda|2009|103236
1|0|1|Ford|2005|134981
2|1|0|Ford|2015|19015
3|2|0|Toyota|2011|73795
4|3|0|Honda|2013|54573

`households: LinkedDataFrame`

household_id|dwelling_type|size
------------|-------------|----
0|house|4
1|apartment|1
2|house|2
3|house|3

Household attributes can be accessed at the vehicle level using the syntax:

```python
vehicles.household.dwelling_type
```

yields:

index|dwelling_type
-----|------------
0|house
1|house
2|apartment
3|house
4|house

Links can go both ways, so to it's possible to aggregate vehicle information about vehicles at the household level. For example, to get the total vehicle-km travelled by a household use:

```python
households.vehicles.sum("km_travelled")
```

yields:

household_id|km_travelled
------------|------------
0|238217
1|19015
2|73795
3|54573

## Constructing

The constructor for `LinkedDataFrame` inherits from `DataFrame` and accepts [the same arguments](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html#pandas.DataFrame). For example, to convert a DataFrame to a LinkedDataFrame, write:

```python
from pandas import DataFrame
from cheval import LinkedDataFrame

...
df: DataFrame
ldf = LinkedDataFrame(df)
...
```

It is also possible to read a LinkedDataFrame directly from file or other external source, using dedicated class methods. These accept all the arguments of the same-named function in the top level Pandas module, e.g. `LinkedDataFrame().read_csv` accepts the same arguments as `pandas.read_csv()`.

The following methods are explicitly wrapped:

```python
from cheval import LinkedDataFrame

LinkedDataFrame.read_csv(*args, **kwargs) -> LinkedDataFrame

LinkedDataFrame.read_table(*args, **kwargs) -> LinkedDataFrame

LinkedDataFrame.read_clipboard(*args, **kwargs) -> LinkedDataFrame

LinkedDataFrame.read_excel(*args, **kwargs) -> LinkedDataFrame

LinkedDataFrame.read_fwf(*args, **kwargs) -> LinkedDataFrame
```

Additionally, `LinkedDataFrame.read_(name, *args, **kwargs)` can be used to access any Pandas read-method that hasn't been made explicit.

## Establishing a Link (Linking two tables)

Once you have a LinkedDataFrame, you can use the `link_to(...)` method to establish a *one-way* link to another frame, assigning it to a new name (referred to as an "alias" in the documentation). For readability it is recommended that the alias be "Pythonic" (e.g. follows rules for Python variable names), but otherwise the link can still be accessed using `[]` syntax (`__getitem__()`).

In order for a link to be established correctly, a LinkedDataFrame needs to know how to relate both tables. To do so, it needs to construct two `Index` objects: one for itself (`self`) and one for the table being linked (`other`). Either of these two can come from the _index_ of the frame or one or more of its _columns_. Most of the args in `link_to()` relate to specifying these indexers. In general, making the join "on" (`on`, `on_self` and `on_other`) refers to **columns**, while making the join using "levels" (`levels`, `self_levels`, `other_levels`) refers to the index. The convenience kwargs `on` and `levels` can be used if the columns or levels have the same name in both frames.

If no indexer args are specified, then the join will be made on all levels of the self AND other indexes. This is the default behaviour.

Multi-indexers are allowed, either from the columns or from the index. Both indexer must always have the same number of levels for the link to be valid; otherwise `LinkageSpecificationError` is raised. Indexing the join using multi-indexes is known to be a slow and memory-intensive operation due to the Pandas' internal representation; it is recommended to pre-compute the indexer (see below) for optimal performance. The RAM- heavy objects will be cleaned by the GC once indexing is complete.

### Pre-computed indexers

By default, when a link is established, the self and other `Index`s are compared and used to pre-compute the _link indexer_. This indexer is a simple NumPy array as long as the `self` table comprising integers from 0 to `len(other)` - in other words, an offset array. This allows for very fast downstream lookups, at the cost of a slow initial linking.

_It is not uncommon for a `link_to()` call to take several seconds, especially when working with `MultiIndex` objects, as it precomputes the fast indexer._ This behaviour can be turned off by passing `precompute=False` when linking, though just delays the problem (as the LinkedDataFrame will just precompute the first time a lookup is requested ).

### Aggregation

LinkedDataFrames test the nature of the relationship upon linking. If there are  multiple matches in the _other_ frame for even one entry in the _self_ frame, then it determines that aggregation is required. `link_to()` returns an enumeration indicating whether or not it determined that aggregation is required for the specified relationship.

### Two-way links

LDF makes no restrictions against two-way links. They can be created by establishing two one-way links between two instances of `LinkedDataFrame`. Users should take care, as in most use cases, one side of the two-way link usually requires aggregation.

## Accessing a linked attribute

In all cases, the final result of a chained attribute lookup return a Series whose index is equal to the leftmost caller. For example, `persons.household.zone.area` returns a Series aligned with `persons`.

### Aggregated Attributes

When LDF has determined that aggregation is required, a reduction operation needs to be specified to return ans attribute. Each reduction operation takes in a string which is passed into `LinkedDataFrame.eval()` for evaluation. This could be a single attribute (e.g. `households.persons.mean("age")` to compute the average age of all the persons in each household) or it can even be a conditional statement (e.g. `households.persons.count("age > 18")` to count the number of adults in each household). It is even possible to access other links inside the aggregation expression.

Some reduction expressions allow for additional `**kwargs`. Under the hood, LinkedDataFrame makes use of the Pandas `DataFrameGroupBy` object, so available keyword arguments can be found on the Pandas website

Currently supported numeric reductions are:

- [`sum(expr="1", *, int_fill=-1)`](https://pandas.pydata.org/pandas-docs/version/0.23.0/generated/pandas.core.groupby.GroupBy.prod.html#pandas.core.groupby.GroupBy.prod)
- [`max(expr="1", *, int_fill=-1)`](https://pandas.pydata.org/pandas-docs/version/0.23.0/generated/pandas.core.groupby.GroupBy.max.html#pandas.core.groupby.GroupBy.max)
- [`min(expr="1", *, int_fill=-1)`](https://pandas.pydata.org/pandas-docs/version/0.23.0/generated/pandas.core.groupby.GroupBy.min.html#pandas.core.groupby.GroupBy.min)
- [`mean(expr="1", *, int_fill=-1)`](https://pandas.pydata.org/pandas-docs/version/0.23.0/generated/pandas.core.groupby.GroupBy.mean.html#pandas.core.groupby.GroupBy.mean)
- [`median(expr="1", *, int_fill=-1)`](https://pandas.pydata.org/pandas-docs/version/0.23.0/generated/pandas.core.groupby.GroupBy.median.html#pandas.core.groupby.GroupBy.median)
- [`prod(expr="1", *, int_fill=-1)`](https://pandas.pydata.org/pandas-docs/version/0.23.0/generated/pandas.core.groupby.GroupBy.prod.html#pandas.core.groupby.GroupBy.prod)
- [`std(expr="1", *, int_fill=-1)`](https://pandas.pydata.org/pandas-docs/version/0.23.0/generated/pandas.core.groupby.GroupBy.std.html#pandas.core.groupby.GroupBy.std)
- [`quantile(expr="1", *, int_fill=-1, q=0.5)`](https://pandas.pydata.org/pandas-docs/version/0.23.0/generated/pandas.core.groupby.DataFrameGroupBy.quantile.html#pandas.core.groupby.DataFrameGroupBy.quantile)
- [`var(expr="1", *, int_fill=-1)`](https://pandas.pydata.org/pandas-docs/version/0.23.0/generated/pandas.core.groupby.GroupBy.var.html#pandas.core.groupby.GroupBy.var)

The following reductions provide additional support for non-numeric data:

- [`count(expr="1")`](https://pandas.pydata.org/pandas-docs/version/0.23.0/generated/pandas.core.groupby.GroupBy.count.html#pandas.core.groupby.GroupBy.count)
- [`first(expr="1")`](https://pandas.pydata.org/pandas-docs/version/0.23.0/generated/pandas.core.groupby.GroupBy.first.html#pandas.core.groupby.GroupBy.first)
- [`last(expr="1")`](https://pandas.pydata.org/pandas-docs/version/0.23.0/generated/pandas.core.groupby.GroupBy.last.html#pandas.core.groupby.GroupBy.last)
- [`nth(expr="1", *, n=1)`](https://pandas.pydata.org/pandas-docs/version/0.23.0/generated/pandas.core.groupby.GroupBy.nth.html#pandas.core.groupby.GroupBy.nth)

## Missing data

LinkedDataFrames support incomplete linkages, filling with a `null`-like or other default/missing value. The default value depends on the `dtype` of the resulting Series. LinkedDataFrame supports all the major types that Pandas supports:

- `int` or `uint` types: 0
- `float` types: `nan`
- `bool`: `False`
- `str`: `""`
- `category`: `nan`
- `datetime`: `nat`
- `object`: `None`

These default values can be modified, either at the class level (affecting all new LinkedDataFrames) or at the instance level (affecting just the selected instance), by calling `set_fill_defaults()`.

It is also possible to specify a fill value for a specific column on a LinkedDataFrame, to allow a more context-appropriate sentinel value. This can be done by calling `set_column_fill()`. Note that doing so only takes effect when the LinkedDataFrame is the _target_ of a link (e.g. passed into the "other" frame)

## Slicing and modifying indexes

When slicing a LinkedDataFrame, links to other tables are only preserved when possible. Because many of Pandas' internal API calls do slicing behind the scenes, it was necessary to silently drop links if they could not be re-established during finalization. In general, slices of rows are safe, and will return a subset of the original LinkedDataFrame with the linkages in-tact. However, operations which slice columns or change the row labels are unsafe when these changes impact links based on such columns or rows. The resulting LinkedDataFrame will have silently discarded any links it could not establish.

### Adding new columns

Adding new columns to targeting (or `other`) frames _even after establishing a link_ is supported. Downstream links will be able to access the new column(s) without any hassle.

### Adding new rows or concatenating

Not supported. The indexers are designed to be fixed; and as it is not possible to expand a NumPy array without making a new one, the same is true for LinkedDataFrames.
