from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# Create word cloud object
wordcloud = WordCloud(width=800, height=800,
                      background_color='white',
                      stopwords=STOPWORDS,
                      min_font_size=10)

# Generate word cloud
wordcloud.generate_from_text(text)

# Create a clickable word cloud
def on_click(word, **kwargs):
    print(word)

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(wordcloud)
ax.axis('off')
wordcloud.recolor(color_func=colors)

# Add interactivity to the word cloud
from matplotlib.widgets import Button
ax_button = plt.axes([0.7, 0.05, 0.1, 0.075])
button = Button(ax_button, 'Click')
button.on_clicked(on_click)

# Show word cloud
plt.show()
