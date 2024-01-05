import pandas as pd
import matplotlib.pyplot as plt
import json

def load_data(file_path):
    return pd.read_csv(file_path)

def load_mappings(json_file_path):
    with open(json_file_path, 'r') as file:
        mappings = json.load(file)
    return mappings['sku_mapping'], mappings['cost_price_mapping']


def calculate_total_quantities(data, sku_mapping):
    total_quantities = {main_sku: 0 for main_sku in sku_mapping}
    for main_sku, variants in sku_mapping.items():
        for variant in variants:
            total_quantities[main_sku] += data[data['product_variant_sku'] == variant]['quantity_sold'].sum()
    return pd.DataFrame(list(total_quantities.items()), columns=['Product Variant SKU', 'Total Quantity Sold'])

def merge_data_with_costs(quantity_table, cost_price_mapping):
    cost_price_df = pd.DataFrame(cost_price_mapping).T.reset_index()
    cost_price_df.columns = ['Product Variant SKU', 'Cost', 'Price']
    return pd.merge(quantity_table, cost_price_df, on='Product Variant SKU', how='left')

def add_ending_quantity(merged_data, data):
    # Filter out rows where 'product_variant_sku' contains 'BUN'
    filtered_data = data[~data['product_variant_sku'].str.contains('BUN', case=False, na=False)]
    # Group by 'product_variant_sku' and filter out those groups with more than one unique 'ending_quantity'
    groups_with_multiple_ending_quantities = filtered_data.groupby('product_variant_sku').filter(lambda x: x['ending_quantity'].nunique() > 1)

    # Print the 'product_variant_sku' values that have multiple 'ending_quantity' entries
    # print("SKUs with multiple ending quantities:")
    # print(groups_with_multiple_ending_quantities['product_variant_sku'].unique())

    # Now perform the grouping and find the minimum 'ending_quantity' on the filtered data
    ending_quantity_per_sku = filtered_data.groupby('product_variant_sku')['ending_quantity'].min().reset_index()
    ending_quantity_per_sku.to_csv('ending_quantity')
    ending_quantity_per_sku.rename(columns={'product_variant_sku': 'Product Variant SKU'}, inplace=True)
    return pd.merge(merged_data, ending_quantity_per_sku, on='Product Variant SKU', how='left')


def plot_quantity_sold_vs_sku(data):
    sorted_data = data.sort_values(by='Total Quantity Sold', ascending=False)
    plt.figure(figsize=(10, 6))
    plt.bar(sorted_data['Product Variant SKU'], sorted_data['Total Quantity Sold'])
    plt.xlabel('Product Variant SKU')
    plt.ylabel('Total Quantity Sold')
    plt.title('SKU vs Total Quantity Sold')
    plt.xticks(rotation=90)
    plt.savefig('sku_vs_total_quantity_sold.png', bbox_inches='tight')
    plt.show()

def order_and_cost(data):
    target_revenue = 200000
    lead_time = 30

    # Calculations
    data['Total Cost'] = (data['Total Quantity Sold'] * data['Cost']).round(0)
    data['Gross Revenue'] = (data['Total Quantity Sold'] * data['Price']).round(0)
    total_current_cost = data['Total Cost'].sum()
    total_current_revenue = data['Gross Revenue'].sum()
    scaling_factor = target_revenue / total_current_revenue if total_current_revenue != 0 else 0
    target_cost = (scaling_factor * total_current_cost).round(0)

    # Add how many need to order
    data['Target Sale Per Unit'] = (data['Total Quantity Sold'] * scaling_factor).round(0)

    # How much to purchase based on the ending quantity and incoming order
    df = pd.read_excel('order.xlsx') 
    incoming_orders = df[['SKU', 'Order Date', 'Order Amount']].dropna(subset=['SKU', 'Order Date'])
    incoming_orders['Order Amount'] = incoming_orders['Order Amount'].fillna(0)    
    incoming_orders.rename(columns={'SKU': 'Product Variant SKU'}, inplace=True)

    data = data.merge(incoming_orders, on='Product Variant SKU', how='left')
    data['Incoming order'] = data['Order Amount'].fillna(0)
    
    # Calculate Purchase Amount
    data['Purchase Amount'] = ((lead_time * data['Target Sale Per Unit'] / 30 - data['ending_quantity'] - data['Incoming order']).clip(lower=0)).round(0)
    data['Real Cost'] = (data['Purchase Amount'] * data['Cost']).round(0)
    real_cost = data['Real Cost'].sum()

    # Display totals
    print(f"Total Current Cost: {total_current_cost}")
    print(f"Total Current Revenue: {total_current_revenue}")
    print(f"Target Revenue: {target_revenue}")
    print(f"Target Cost: {target_cost}")
    print(f"Real Cost: {real_cost}")

    data.to_csv(f"order_and_revenue_{target_revenue}.csv")

    return data




def plot_projected_ending_quantity(data):
    df = pd.read_excel('order.xlsx') 
    # Extract relevant columns
    # Assuming 'Variant' corresponds to 'Product Variant SKU', 'Order Date' to 'date_arrive', and 'Order Amount' to 'order_quantity'
    incoming_orders = df[['SKU', 'Order Date', 'Order Amount']].dropna()

    # Rename columns to match your existing DataFrame structure
    incoming_orders.columns = ['Product Variant SKU', 'date_arrive', 'order_quantity']

    # Convert 'date_arrive' to datetime
    incoming_orders['date_arrive'] = pd.to_datetime(incoming_orders['date_arrive'])


    # Set the start date, number of months for the projection, and threshold
    start_date = '2024-01-04'
    months_for_projection = 3
    inventory_threshold = 2  # Months of inventory before considering it critical
    days_in_month = 30
    total_days = months_for_projection * days_in_month

    # Generate date range
    date_range = pd.date_range(start=start_date, periods=total_days)

    # Calculate daily sales rate
    data['Daily Sales Rate'] = data['Total Quantity Sold'] / days_in_month
    data['Months of Inventory'] = data['ending_quantity'] / data['Total Quantity Sold']

    # Filter SKUs based on the threshold
    filtered_data = data[data['Months of Inventory'] < inventory_threshold]

    # Initialize a dictionary for time series data
    time_series_data = {}

    # Calculate projected ending quantity over time, considering incoming orders
    for index, row in filtered_data.iterrows():
        sku = row['Product Variant SKU']
        daily_sales = row['Daily Sales Rate']
        ending_quantity = row['ending_quantity']
        
        quantities = []
        for day in date_range:
            # Check if there is an incoming order for this SKU on this day
            order_row = incoming_orders[(incoming_orders['Product Variant SKU'] == sku) & (incoming_orders['date_arrive'] == day)]
            if not order_row.empty:
                ending_quantity += order_row['order_quantity'].values[0]
            
            # Calculate the projected ending quantity
            projected_quantity = max(ending_quantity - daily_sales, 0)
            quantities.append(projected_quantity)
            
            # Update the ending quantity for the next day
            ending_quantity = projected_quantity

        time_series_data[sku] = quantities

    # Convert the dictionary to a DataFrame
    time_series_df = pd.DataFrame(time_series_data, index=date_range)

    # Plotting
    plt.figure(figsize=(15, 8))
    for sku in time_series_data.keys():
        plt.plot(time_series_df.index, time_series_df[sku], label=sku)

    plt.xlabel('Date')
    plt.ylabel('Projected Ending Quantity')
    plt.title('SKU Projected Ending Quantity Over Time with Incoming Orders')
    plt.legend()
    # plt.show()
    # time_series_df.to_csv("projected_inventory.csv")


# Main execution
file_path = 'inventory_sales_2023-12-01_2023-12-31.csv'
data = load_data(file_path)
[sku_mapping, cost_price_mapping] = load_mappings('mapping.json')

quantity_table = calculate_total_quantities(data, sku_mapping)
merged_data = merge_data_with_costs(quantity_table, cost_price_mapping)
data = add_ending_quantity(merged_data, data)
# plot_projected_ending_quantity(data)

# plot_quantity_sold_vs_sku(data)
order_and_cost(data)



