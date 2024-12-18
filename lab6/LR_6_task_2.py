import pandas as pd

def calculate_likelihood(dataframe, feature, feature_value):
    positive_likelihood = len(
        dataframe[(dataframe[feature] == feature_value) & (dataframe["Play"] == "Yes")]
    ) / len(dataframe[dataframe["Play"] == "Yes"])

    negative_likelihood = len(
        dataframe[(dataframe[feature] == feature_value) & (dataframe["Play"] == "No")]
    ) / len(dataframe[dataframe["Play"] == "No"])

    return positive_likelihood, negative_likelihood

def calculate_combined_probability(*probabilities):
    combined_probability = 1
    for probability in probabilities:
        combined_probability *= probability

    return combined_probability

data = pd.DataFrame(
    {
        "Day": [
            "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11", "D12", "D13", "D14",
        ],
        "Outlook": [
            "Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain", "Overcast", "Sunny", "Sunny", "Rain", "Sunny", "Overcast", "Overcast", "Rain",
        ],
        "Humidity": [
            "High", "High", "High", "High", "Normal", "Normal", "Normal", "High", "Normal", "Normal", "Normal", "High", "Normal", "High",
        ],
        "Wind": [
            "Weak", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Strong",
        ],
        "Play": [
            "No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No",
        ],
    }
)

selected_outlook = "Sunny"
selected_humidity = "High"
selected_wind = "Weak"

outlook_yes_likelihood, outlook_no_likelihood = calculate_likelihood(data, "Outlook", selected_outlook)
humidity_yes_likelihood, humidity_no_likelihood = calculate_likelihood(data, "Humidity", selected_humidity)
wind_yes_likelihood, wind_no_likelihood = calculate_likelihood(data, "Wind", selected_wind)

positive_probability = calculate_combined_probability(
    outlook_yes_likelihood,
    humidity_yes_likelihood,
    wind_yes_likelihood,
)
negative_probability = calculate_combined_probability(
    outlook_no_likelihood,
    humidity_no_likelihood,
    wind_no_likelihood,
)

total_probability = positive_probability + negative_probability
probability_yes = positive_probability / total_probability
probability_no = negative_probability / total_probability

print(f"Match WILL happen: {probability_yes:.2f};")
print(f"Match will NOT happen: {probability_no:.2f};")
