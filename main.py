import pandas as pd
from dotenv import load_dotenv


def calculate_weighted_score(weights, product_data):
    product_data['weighted_score'] = (
            product_data['profitability'] * weights['profitability'] +
            product_data['market_demand'] * weights['market_demand'] +
            product_data['competition'] * weights['competition'] +
            product_data['differentiation'] * weights['differentiation'] +
            product_data['sourcing'] * weights['sourcing']
    )
    return product_data


def main():
    load_dotenv()
    weights = {
        "profitability": float(os.getenv('PROFITABILITY_WEIGHT') or '0'),
        "market_demand": float(os.getenv('MARKET_DEMAND_WEIGHT') or '0'),
        "competition": float(os.getenv('COMPETITION_WEIGHT')),
        "differentiation": float(os.getenv('DIFFERENTIATION_WEIGHT')),
        "sourcing": float(os.getenv('SOURCING_WEIGHT'))
    }
    product_data = pd.read_csv('product_data.csv')
    ranked_products = calculate_weighted_score(weights, product_data)
    ranked_products = ranked_products.sort_values(by='weighted_score', ascending=False)
    print(ranked_products)
