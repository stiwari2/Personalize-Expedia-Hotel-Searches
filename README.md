## Kaggle - Personalize Expedia Hotel Searches - ICDM 2013 ##


Dataset- The Kaggle dataset is split into a training and a test set from the original dataset. Essentially, the dataset contains information about a search query of a user for a hotel, the hotel properties that resulted and for the training set, whether the user clicked on the hotel and booked it. 

1) File descriptions\
training_set - the training set\
test_set - the test set\
submission_sample - a sample submission file in the correct format


2) Data fields\
Please refer to https://www.kaggle.com/c/expedia-personalized-sort/data \
Each line in the dataset represents a combination of a search query by a user with one specific hotel property that was shown as part of the results\
A list of hotels is presented to the user\
Lines that belong to the same user/search are identified by the same search id\

Training set\
**position** - (Integer) - Hotel position on Expedia's search results page. This is only provided for the training data, but not the test data\
**click_bool** - (Boolean) - 1 if the user clicked on the property, 0 if not\
**booking_bool** - (Boolean) - 1 if the user booked the property, 0 if not\
**gross_booking_usd** - (Float) - Total value of the transaction. This can differ from the price_usd due to taxes, fees, conventions on multiple day bookings and purchase of a room type other than the one shown in the search\
