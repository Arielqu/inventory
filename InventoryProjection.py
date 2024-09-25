import pandas as pd
import matplotlib.pyplot as plt
import json
import os

key = '09-24'
config = {
    'start_date': '2024-09-24',
    'months_for_projection': 3,
    'inventory_threshold': 90,  # Days of inventory before considering it critical
    'days_in_range': 30,
    'target_revenue': 100000,
    'lead_time': 60, #how many days it take to get the product
    'print_level': 2,  # 0: Nothing 1. Print and save 2. Plot
    'input_data_path':f"inventory_sales_2024-08-19_2024-09-17.csv",
    'new_data_path':f"InventoryQuantitiesByWarehouse-9-24.csv",
    'recent_sales_data_path': f'inventory_sales_2024-09-11_2024-09-17.csv',
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
    filtered_data = data[~data['product_variant_sku'].str.contains('BUN', case=False, na=False)]
    ending_quantity_per_sku = filtered_data.groupby('product_variant_sku')['ending_quantity'].min().reset_index()
    ending_quantity_per_sku.rename(columns={'product_variant_sku': 'Product Variant SKU'}, inplace=True)
    return pd.merge(merged_data, ending_quantity_per_sku, on='Product Variant SKU', how='left')

def replace_ending_quantity(original_data, new_data_path):
    new_data = pd.read_csv(new_data_path)
    new_data = new_data[new_data['Primary Supplier'] == 'FreezBone']
    # new_data.rename(columns={'SKU': 'Product Variant SKU', 'Available Quantity': 'new_ending_quantity'}, inplace=True)
    new_data.rename(columns={'SKU': 'Product Variant SKU', 'Total': 'new_ending_quantity'}, inplace=True)
    
    missing_skus = []
    
    for index, row in original_data.iterrows():
        sku = row['Product Variant SKU']
        if sku in new_data['Product Variant SKU'].values:
            new_quantity = new_data.loc[new_data['Product Variant SKU'] == sku, 'new_ending_quantity'].values[0]
            original_data.at[index, 'ending_quantity'] = new_quantity
        else:
            missing_skus.append(sku)
    
    if missing_skus:
        print("The following SKUs were not found in the new data:")
        for missing_sku in missing_skus:
            print(missing_sku)
    
    return original_data

def deduct_recent_sale(data):
    recent_sales_data = pd.read_csv(os.path.join('inputs',config['recent_sales_data_path']))
    
    recent_sales_totals = recent_sales_data.groupby('product_variant_sku')['quantity_sold'].sum().reset_index()
    recent_sales_totals.rename(columns={'product_variant_sku': 'Product Variant SKU', 'quantity_sold': 'Recent Quantity Sold'}, inplace=True)
    
    data = pd.merge(data, recent_sales_totals, on='Product Variant SKU', how='left')
    data['Recent Quantity Sold'] = data['Recent Quantity Sold'].fillna(0)
    data['ending_quantity'] -= data['Recent Quantity Sold']
    
    return data

def order_and_cost(data):
    data['Total Cost'] = (data['Total Quantity Sold'] * data['Cost']).round(0)
    data['Gross Revenue'] = (data['Total Quantity Sold'] * data['Price']).round(0)
    total_current_cost = data['Total Cost'].sum().round(0)
    total_current_revenue = data['Gross Revenue'].sum().round(0)
    
    scaling_factor = (config['target_revenue'] / total_current_revenue).round(1) if total_current_revenue != 0 else 0
    target_cost = (scaling_factor * total_current_cost).round(0)

    data['Target Sale Per Unit'] = (data['Total Quantity Sold'] * scaling_factor).round(0)

    df = pd.read_excel('inputs//order.xlsx') 
    incoming_orders = df[['SKU', 'Order Date', 'Order Amount']].dropna(subset=['SKU', 'Order Date'])

    incoming_orders = incoming_orders.groupby('SKU')['Order Amount'].sum().reset_index().round(0)
    incoming_orders.rename(columns={'SKU': 'Product Variant SKU'}, inplace=True)

    data = data.merge(incoming_orders, on='Product Variant SKU', how='left')
    data['Incoming order'] = data['Order Amount'].fillna(0).round(0)
    
    data['Purchase Amount'] = ((config['lead_time'] * data['Target Sale Per Unit'] / 30 - data['ending_quantity'] - data['Incoming order']).clip(lower=0)).round(0)
    data['Real Cost'] = (data['Purchase Amount'] * data['Cost']).round(0)
    real_cost = data['Real Cost'].sum().round(0)

    if config['print_level'] > 0:
        print(f"Total Current Cost: {total_current_cost}")
        print(f"Total Current Revenue: {total_current_revenue}")
        print(f"Target Revenue: {config['target_revenue']}")
        print(f"Target Cost: {target_cost}")
        print(f"Real Cost: {real_cost}")
        data.to_csv(f"{key}//{config['order_and_cost_file_name']}_{config['target_revenue']}.csv", index=False)

    return data

def plot_projected_ending_quantity(data):
    df = pd.read_excel('inputs//order.xlsx') 
    incoming_orders = df[['SKU', 'Order Date', 'Order Amount']].dropna()
    incoming_orders.columns = ['Product Variant SKU', 'date_arrive', 'order_quantity']
    incoming_orders['date_arrive'] = pd.to_datetime(incoming_orders['date_arrive'])
    total_days = config['months_for_projection'] * 30
    date_range = pd.date_range(start=config['start_date'], periods=total_days)
    data['Daily Sales Rate'] = (data['Total Quantity Sold'] / config['days_in_range']).round(0)
    data['Days of Inventory'] = (data['ending_quantity'] / data['Daily Sales Rate']).round(0)
    filtered_data = data[data['Days of Inventory'] < config['inventory_threshold']]
    time_series_data = {}
    for index, row in filtered_data.iterrows():
        sku = row['Product Variant SKU']
        daily_sales = row['Daily Sales Rate']
        ending_quantity = row['ending_quantity']
        quantities = []
        for day in date_range:
            order_row = incoming_orders[(incoming_orders['Product Variant SKU'] == sku) & (incoming_orders['date_arrive'] == day)]
            if not order_row.empty:
                ending_quantity += order_row['order_quantity'].values[0]
            projected_quantity = max(ending_quantity - daily_sales, 0)
            quantities.append(projected_quantity)
            ending_quantity = projected_quantity
        time_series_data[sku] = quantities
    time_series_df = pd.DataFrame(time_series_data, index=date_range)
    if (config['print_level']>1):
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

# Load data and mappings
data = load_data(os.path.join('inputs',config['input_data_path']))
[sku_mapping, cost_price_mapping] = load_mappings(os.path.join('inputs','mapping.json'))

# Calculate quantities and costs
quantity_table = calculate_total_quantities(data, sku_mapping)
merged_data = merge_data_with_costs(quantity_table, cost_price_mapping)
data = add_ending_quantity(merged_data, data)
data = replace_ending_quantity(merged_data, os.path.join('inputs',config['new_data_path']))
# data = deduct_recent_sale(data)

# Plot and analyze data
plot_projected_ending_quantity(data)
final_data = order_and_cost(data)

