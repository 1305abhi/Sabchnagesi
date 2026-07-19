from reportlab.pdfgen import canvas
import sys

def create_pdf(filename, text):
    c = canvas.Canvas(filename)
    y = 800
    # Split text into lines that fit on a page
    words = text.split()
    lines = []
    current_line = []
    for word in words:
        if len(" ".join(current_line + [word])) * 6 > 400: # Simple approximation for width
            lines.append(" ".join(current_line))
            current_line = [word]
        else:
            current_line.append(word)
    if current_line:
        lines.append(" ".join(current_line))
        
    for line in lines:
        if y < 50:
            c.showPage()
            y = 800
        c.drawString(72, y, line)
        y -= 18
    c.save()
    print(f"Created PDF: {filename}")

if __name__ == '__main__':
    human_text = (
        "Yesterday, I decided to go for a long walk in the nearby park to clear my head after a busy week of coding. "
        "The weather was absolutely perfect – crisp, cool autumn air and a bright blue sky without a single cloud. "
        "As I walked along the leaf-strewn paths, I watched a family of ducks swimming lazily in the pond and listened "
        "to the rustling of the oak trees in the gentle breeze. It's funny how a simple hour in nature can completely "
        "restore your energy and put things into perspective. I came back home feeling inspired and ready to tackle "
        "new engineering challenges."
    )
    create_pdf("test_human.pdf", human_text)
    
    ai_text = (
        "Artificial Intelligence (AI) represents one of the most transformative technology developments of the twenty-first century. "
        "By enabling computer systems to perform complex tasks that historically required human cognitive capabilities—such as visual "
        "perception, speech recognition, decision-making, and translation between languages—AI is reshaping global industries. "
        "Large language models, a subset of generative AI, utilize deep learning neural networks trained on vast datasets to "
        "synthesize text, write software code, and analyze structured data with remarkable speed and coherence. "
        "As these technologies continue to advance, standard methodologies in academic research, education, and software development "
        "are undergoing a fundamental paradigm shift."
    )
    create_pdf("test_ai.pdf", ai_text)
