import pandas as pd
import random
from sklearn.naive_bayes import MultinomialNB
import re
from sklearn.feature_extraction.text import CountVectorizer

model = MultinomialNB()
vectorizer = CountVectorizer()

df = pd.read_csv("Data/chat.csv")
# Preprocess text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

# To apply a function to your dataset use the .apply() keyword
df["text"] = f["text"].apply(clean_text)

#   AI Create a dictionary and Convert words to numbers
X = vectorizer.fit_transform(df["text"])

# AI Model learns patterns for classfication

model.fit(X, df["intent"])

responses = {

    "spam": [
        "⚠️ Warning! Dude, that looks like spam. Don’t trust it.",
        "🚫 Be careful! This message is likely spam. Avoid clicking anything.",
        "⚠️  Spam detected! You should delete this immediately.",
        "🚨 That’s suspicious. It strongly looks like a spam message.",
        "❌ Do NOT click it. This is probably a scam or spam.",
        "⚠️ Red flag! This message is unsafe and may be spam.",
        "🚫 Delete this message. It is not trustworthy.",
        "⚠️ That message has spam patterns. Stay alert.",
        "🚨 I strongly advise you not to interact with this message.",
        "❌ This looks like a phishing or spam attempt.",
        "⚠️ Be careful! This is not a normal message.",
        "🚫 Spam alert! Ignore and remove it.",
        "⚠️ This message may try to trick you. Stay safe.",
        "❌ High chance of spam. Don’t respond to it.",
        "🚨 Suspicious content detected. Treat it as spam."
    ],

    "ham": [
        "✅ This looks safe. Just a normal message.",
        "👍 No worries, this is not spam.",
        "📩 This is a regular message from a real conversation.",
        "🙂 Safe message detected. You’re fine.",
        "👌 This appears to be normal communication.",
        "🟢 No spam here. It’s safe to read.",
        "📨 Just a casual message, nothing suspicious.",
        "😊 This looks like a friend or normal sender.",
        "✔️ Everything seems fine with this message.",
        "💬 Normal message detected. No action needed.",
        "🧠 This is safe and legitimate content.",
        "📱 Looks like a genuine message.",
        "😌 Nothing suspicious here, you can trust it.",
        "🟢 Safe message. No risk detected.",
        "👌 This is just regular communication."
    ]
}
print("Hello, I'm a simple AI chatbot for predicting Spam in Emails")
print("Just copy and paste the exact words on your email here.")
# print("")

while True:
    user_input = input("You: ")
    if user_input == "bye":
        print("Goodbye, Have a great day. I'm always happy to help")
        break
    X_test = vectorizer.transform([user_input])
    # Bot classifies user_input (Spam Or Ham)
    intent = model.predict(X_test)[0]
    # Model then Replies
    if intent in responses:
        print("Bot: ", random.choice(responses[intent]))
    else:
        print("I'm still figuring that out.")
        



