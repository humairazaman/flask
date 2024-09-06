import torch
import torch.nn as nn
import joblib

# Define the Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, nhead=6, num_layers=3, d_model=512, dropout=0.3):
        super(TransformerModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            dropout=dropout
        )
        self.fc2 = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = x.permute(1, 0, 2)
        output = self.transformer(x, x)
        output = output.mean(dim=0)
        output = self.fc2(output)
        return output


# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=512, num_layers=3, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take the output of the last time step
        x = self.fc(x)
        return x


# Define the Ensemble Model
class EnsembleModel(nn.Module):
    def __init__(self, transformer_model, lstm_model, num_classes):
        super(EnsembleModel, self).__init__()
        self.transformer_model = transformer_model
        self.lstm_model = lstm_model
        self.fc = nn.Linear(num_classes * 2, num_classes)  # Combine outputs from both models

    def forward(self, x):
        transformer_output = self.transformer_model(x)
        lstm_output = self.lstm_model(x)
        combined_output = torch.cat((transformer_output, lstm_output), dim=1)
        output = self.fc(combined_output)
        return output


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Function to load model and scaler
def load_model_and_scaler(model_path, scaler_path, num_classes):
    # Instantiate the Transformer and LSTM models
    transformer_model = TransformerModel(
        input_dim=1662,
        num_classes=num_classes,
        nhead=8,
        num_layers=4,
        d_model=512,
        dropout=0.3
    ).to(device)

    lstm_model = LSTMModel(
        input_dim=1662,
        num_classes=num_classes,
        hidden_dim=512,
        num_layers=3,
        dropout=0.3
    ).to(device)

    # Instantiate the Ensemble model with both Transformer and LSTM models
    model = EnsembleModel(
        transformer_model=transformer_model,
        lstm_model=lstm_model,
        num_classes=num_classes
    ).to(device)

    # Load the saved model parameters
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)

    model.eval()

    # Load the scaler
    scaler = joblib.load(scaler_path)

    return model, scaler


# Define all categories
categories = {
    'greeting': {
        'model_path': 'model_greetings.pth',
        'scaler_path': 'greetingscaler.pkl',
        'actions': [
            "السلام وعلیکم",
            "صبح بخیر",
            "ایک اچھا دن گزاریں",
            "بعد میں ملتے ہیں",
            "خوش آمدید"
        ]
    },
    'daily_routine': {
        'model_path': 'model_everyday.pth',
        'scaler_path': 'everydaycaler.pkl',
        'actions': [
            "ایمبولینس کو کال کریں",
            "کیا میں آپ کا حکم لے سکتا ہوں؟",
            "میں بیمار ہوں",
            "میں نے پوری رات مطالعہ کیا",
            "چلو ایک ریستوراں میں چلو"
        ]
    },
    'question': {
        'model_path': 'model_question.pth',
        'scaler_path': 'questionscaler.pkl',
        'actions': [
            "کیا تم بھوکے ہو؟",
            "آپ کیسے ہیں؟",
            "اس کی کیا قیمت ہے؟",
            "میں نہیں سمجھا",
            "آپ کا ٹیلیفون نمبر کیا ہے؟"
        ]
    }
}


# Initialize models and scalers for all categories
def initialize_categories():
    initialized_categories = {}
    for category_name, data in categories.items():
        # Load model and scaler for each category
        model, scaler = load_model_and_scaler(
            model_path=data['model_path'],
            scaler_path=data['scaler_path'],
            num_classes=len(data['actions'])
        )
        initialized_categories[category_name] = {
            'model': model,
            'scaler': scaler,
            'actions': data['actions']
        }
    return initialized_categories


# Initialize and store in a variable
initialized_categories = initialize_categories()




