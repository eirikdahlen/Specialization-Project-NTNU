import os
import pandas as pd
import numpy as np
import time
import tensorflow as tf
from transformers import BertTokenizerFast, TFBertForSequenceClassification, BertConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

start = time.time()

df = pd.read_csv("frikk_eirik_dataset.csv")

X, y = df['text_document'], df["target"]

X_a, X_b, y_a, y_b = train_test_split(X, y, train_size=0.8, random_state=42, stratify=df['target'])

X_list = X_a.values.tolist()
for i in range(len(X_list)):
    X_list[i] = str(X_list[i])
X_val = X_b.values.tolist()
for i in range(len(X_val)):
    X_val[i] = str(X_val[i])


y_list = y_a.values.tolist()
labels_dict = {'unrelated': 0, 'pro_ed': 1, 'pro_recovery': 2}
for i in range(len(y_list)):
    y_list[i] = labels_dict[y_list[i]]


y_val = y_b.values.tolist()
for i in range(len(y_val)):
    y_val[i] = labels_dict[y_val[i]]

time_a = time.time() - start

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(X_list, truncation=True, padding=True)
time_b = time.time() - time_a - start
print(f"Created train encodings, time used {time_b}")
val_encodings = tokenizer(X_val, truncation=True, padding=True)
time_c = time.time() - time_b - time_a - start
print(f"Created val encodings, time used {time_c}")

train_dataset = np.array(list(dict(train_encodings).values()))
val_dataset = np.array(list(dict(val_encodings).values()))

BATCH_SIZE = 16

# Create a callback that saves the model's weights every x epochs
checkpoint_path = "bert16_ckpt/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True)

save_model = True


config = BertConfig(num_labels=3, return_dict=True, model_type='bert-base-uncased')

model = TFBertForSequenceClassification(config=config)

if save_model:
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=['accuracy'])

    model.fit(
        train_dataset[0],
        np.array(y_list),
        epochs=5,
        batch_size=BATCH_SIZE,
        callbacks=[cp_callback]
        )
else:
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)

preds = model.predict(val_dataset[0])["logits"]

preds_proba = tf.keras.backend.softmax(preds, axis=1)

classes = np.argmax(preds, axis=-1)

score = classification_report(y_val, classes, digits=3)
print(score)

total = time.time()  - start
print(f"Done in: {total}")  
