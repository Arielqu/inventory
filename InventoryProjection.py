import pandas as pd
import matplotlib.pyplot as plt
import json
import os
key = '08-20-MORE'
config = {
    'start_date': '2024-08-20',
    'months_for_projection': 3,
    'inventory_threshold': 90,  # Days of inventory before considering it critical
    'days_in_range': 90,
    'target_revenue': 300000,
    'lead_time': 30, #how many days it take to get the product
    'print_level': 2,  # 0: Nothing 1. Print and save 2. Plot
    'input_data_path':f"inventory_sales_2024-05-22_2024-08-19.csv",
    'new_data_path':f"TotalAvailableInventory.csv",
    'order_and_cost_file_name':f'order_and_cost_file_name_{key}',
    'projection_file_name':f'projection_{key}'
}

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

def replace_ending_quantity(original_data, new_data_path):
    # Load the new Excel file
    new_data = pd.read_csv(new_data_path)

    # Filter the new data to only include rows where "Primary Supplier" is "FreezBone"
    new_data = new_data[new_data['Primary Supplier'] == 'FreezBone']
    
    
    # Ensure the SKU column in the new data matches with the Product Variant SKU in the original data
    new_data.rename(columns={'SKU': 'Product Variant SKU', 'Available Quantity': 'new_ending_quantity'}, inplace=True)
    
    # Initialize a list for missing SKUs
    missing_skus = []
    
    # Iterate through each row in the original data
    for index, row in original_data.iterrows():
        sku = row['Product Variant SKU']
        
        # Check if this SKU exists in the new data
        if sku in new_data['Product Variant SKU'].values:
            # If found, update the ending_quantity with the new value
            new_quantity = new_data.loc[new_data['Product Variant SKU'] == sku, 'new_ending_quantity'].values[0]
            original_data.at[index, 'ending_quantity'] = new_quantity
        else:
            # If not found, log the missing SKU
            missing_skus.append(sku)
    
    # Print out the SKUs that were not found in the new data
    if missing_skus:
        print("The following SKUs were not found in the new data:")
        for missing_sku in missing_skus:
            print(missing_sku)
    
    return original_data


def plot_quantity_sold_vs_sku(data):
    sorted_data = data.sort_values(by='Total Quantity Sold', ascending=False)
    plt.figure(figsize=(10, 6))
    plt.bar(sorted_data['Product Variant SKU'], sorted_data['Total Quantity Sold'])
    plt.xlabel('Product Variant SKU')
    plt.ylabel('Total Quantity Sold')
    plt.title('SKU vs Total Quantity Sold')
    plt.xticks(rotation=90)
    if (config['print_level'] > 0):
        plt.savefig(f'{key}//sku_vs_total_quantity_sold.png', bbox_inches='tight')
    if (config['print_level'] > 1):
        plt.show()

def order_and_cost(data):
    # Calculations
    data['Total Cost'] = (data['Total Quantity Sold'] * data['Cost']).round(0)
    data['Gross Revenue'] = (data['Total Quantity Sold'] * data['Price']).round(0)
    total_current_cost = data['Total Cost'].sum()
    total_current_revenue = data['Gross Revenue'].sum()
    scaling_factor = config['target_revenue']/ total_current_revenue if total_current_revenue != 0 else 0
    target_cost = (scaling_factor * total_current_cost).round(0)

    # Add how many need to order
    data['Target Sale Per Unit'] = (data['Total Quantity Sold'] * scaling_factor).round(0)

    # How much to purchase based on the ending quantity and incoming order
    df = pd.read_excel('inputs//order.xlsx') 
    incoming_orders = df[['SKU', 'Order Date', 'Order Amount']].dropna(subset=['SKU', 'Order Date'])
    incoming_orders['Order Amount'] = incoming_orders['Order Amount'].fillna(0)    
    incoming_orders.rename(columns={'SKU': 'Product Variant SKU'}, inplace=True)

    data = data.merge(incoming_orders, on='Product Variant SKU', how='left')
    data['Incoming order'] = data['Order Amount'].fillna(0)
    
    # Calculate Purchase Amount
    data['Purchase Amount'] = ((config['lead_time'] * data['Target Sale Per Unit']/30 - data['ending_quantity'] - data['Incoming order']).clip(lower=0)).round(0)
    data['Real Cost'] = (data['Purchase Amount'] * data['Cost']).round(0)
    real_cost = data['Real Cost'].sum()

    if (config['print_level']>0):
        # Display totals
        print(f"Total Current Cost: {total_current_cost}")
        print(f"Total Current Revenue: {total_current_revenue}")
        print(f"Target Revenue: { config['target_revenue']}")
        print(f"Target Cost: {target_cost}")
        print(f"Real Cost: {real_cost}")
        data.to_csv(f"{key}//{config['order_and_cost_file_name']}_{ config['target_revenue']}.csv")
    return data


def plot_projected_ending_quantity(data):
    df = pd.read_excel('inputs//order.xlsx') 
    # Extract relevant columns
    # Assuming 'Variant' corresponds to 'Product Variant SKU', 'Order Date' to 'date_arrive', and 'Order Amount' to 'order_quantity'
    incoming_orders = df[['SKU', 'Order Date', 'Order Amount']].dropna()
    # Rename columns to match your existing DataFrame structure
    incoming_orders.columns = ['Product Variant SKU', 'date_arrive', 'order_quantity']

    # Convert 'date_arrive' to datetime
    incoming_orders['date_arrive'] = pd.to_datetime(incoming_orders['date_arrive'])

    # Set the start date, number of months for the projection, and threshold

    total_days = config['months_for_projection'] * 30

    # Generate date range
    date_range = pd.date_range(start=config['start_date'], periods=total_days)

    # Calculate daily sales rate
    data['Daily Sales Rate'] = data['Total Quantity Sold'] / config['days_in_range']
    data['Days of Inventory'] = data['ending_quantity'] / data['Daily Sales Rate']

    # Filter SKUs based on the threshold
    filtered_data = data[data['Days of Inventory'] < config['inventory_threshold']]

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
    if (config['print_level']>1):
        # Plotting
        plt.figure(figsize=(15, 8))
        for sku in time_series_data.keys():
            plt.plot(time_series_df.index, time_series_df[sku], label=sku)
        plt.xlabel('Date')
        plt.ylabel('Projected Ending Quantity')
        plt.title('SKU Projected Ending Quantity Over Time with Incoming Orders')
        plt.legend()
        plt.show()
        plt.savefig(f"{key}//{config['projection_file_name']}.png", bbox_inches='tight')
    if (config['print_level']>0):    
        time_series_df.to_csv(f"{key}//{config['projection_file_name']}.csv")

# Main execution
if not os.path.exists(key):
    os.makedirs(key)
data = load_data(os.path.join('inputs',config['input_data_path']))
[sku_mapping, cost_price_mapping] = load_mappings(os.path.join('inputs','mapping.json'))
quantity_table = calculate_total_quantities(data, sku_mapping)
merged_data = merge_data_with_costs(quantity_table, cost_price_mapping)
data = add_ending_quantity(merged_data, data)
data = replace_ending_quantity(merged_data, os.path.join('inputs',config['new_data_path']))
plot_quantity_sold_vs_sku(data)
plot_projected_ending_quantity(data)
order_and_cost(data)



