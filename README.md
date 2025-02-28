This code is a smart Telegram bot designed for product search and car part image recognition. The bot consists of two main components:

Data Processing: Using the DataProcessor class, product data is read from an Excel file and preprocessed (e.g., normalizing Persian text), enabling users to search for products by name.

Image Recognition: A Convolutional Neural Network (CNN) model, implemented in the CNNModel class, allows the bot to recognize car parts from images sent by users. If the model is already trained, it is loaded; otherwise, the model is trained and saved.

The bot is activated with the /start command, and users can either search for a product by name or send an image of a car part for the bot to identify. This code is suitable for projects involving image processing and product search.
