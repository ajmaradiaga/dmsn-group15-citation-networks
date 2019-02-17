from bokeh.io import output_notebook
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
from bokeh.models import ColumnDataSource
from bokeh.plotting import show

output_notebook()

def display_df_with_bokeh(df, columns=None, range_of_records=slice(0,20), include_index=False):
    
    table_columns = []
    
    if columns is None:
        columns = {column: column for column in df}
    
    if range_of_records is not None:
        df = df[range_of_records]
    
    if include_index:
        table_columns.append(TableColumn(field=df.index.name, title=df.index.name))
    
    for field, title in columns.items():
        table_columns.append(TableColumn(field=field, title=title))
    
    
    data_table = DataTable(columns=table_columns, source=ColumnDataSource(df)) # bokeh table

    show(data_table)