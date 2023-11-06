import spacy
from spacy.util import minibatch, compounding

nlp = spacy.load("en_core_web_sm")

title = "10 things your boss does not want you to know"

intro = """Have you ever wondered what your boss might be hiding from you? In this article, we will uncover the top 10 things that your boss may not want you to know. Whether it's about company policies, work dynamics, or hidden agendas, it's essential to stay informed. Let's dive right in!"""

key_points = [
    "The real company financial situation",
    "Internal office politics and power struggles",
    "Actual reasons behind key business decisions",
    "Information about confidential projects and clients",
    "Insider knowledge about upcoming layoffs or restructurings",
    "True motives behind sudden changes in company policies",
    "Hidden conflicts within the management team",
    "Unspoken expectations for promotions and salary raises",
    "Sneaky tactics used to keep employees in the dark",
    "Behind-the-scenes strategies to maintain a competitive edge",
]

def generate_blog_article(title, intro, key_points):
    article = f"# {title}\n\n"
    article += intro + "\n\n"
    for i, point in enumerate(key_points, start=1):
        article += f"## {i}. {point}\n\n"
        doc = nlp(point)
        generated_text = ""
        for _ in range(3):  # Generate 3 sentences for each key point
            sentence = " ".join([token.text for token in doc])
            generated_text += sentence.capitalize() + ". "
        article += generated_text + "\n\n"
    return article

blog_article = generate_blog_article(title, intro, key_points)
print(blog_article)
