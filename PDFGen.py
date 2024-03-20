import reportlab
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
reportlab.rl_config.TTFSearchPath.append('c:/winnt/fonts'+'c:/windows/fonts')
pdfmetrics.registerFont(TTFont('Arial', 'Arial.ttf'))
pdfmetrics.registerFont(TTFont('Arial Bold', 'arialbd.ttf'))
pdfmetrics.registerFont(TTFont('Arial Narrow Bold', 'ARIALNB.ttf'))
from itertools import zip_longest
import random
import string

def add_back_image(pdf_canvas, image_url):
    # Get the page size
    page_width, page_height = pdf_canvas._pagesize
    # Add the resized (and possibly flipped) image to the canvas
    pdf_canvas.drawImage(image_url, 0, 0, width=page_width, height=page_height)
    # Restore the canvas state after flipping
        
        
def add_horizontal_line(pdf_canvas, line_x_position, y_position, line_length):
    pdf_canvas.line(line_x_position, y_position, line_x_position+line_length, y_position)

def generate_pdf(output_arr1, output_arr2, output_file_path):
    # PDF canvas setup
    pdf_canvas = canvas.Canvas(output_file_path)

    # Specify the image URL
    image_url = "https://img.freepik.com/free-vector/pastel-gradient-blur-vector-background_53876-174858.jpg"###BASE
    image_url = "https://img.freepik.com/free-vector/blank-rectangle-black-abstract-frame_53876-118331.jpg"
    image_url = "https://img.freepik.com/free-photo/marble-textured-backdrop-frame_53876-90102.jpg"
    image_url = "assets/PDF-BG.jpg"

    # Set the desired width and height for the image
    image_width = 600
    image_height = 300

    # Initial coordinates
    x_coordinate = 120
    y_coordinate = 700

    # Maximum y-coordinate before adding a new page
    max_y_coordinate = 100

    image_direction = True
    count_page = 2

    maxone = len(output_arr1)>len(output_arr2)
    # Add the centered image to the PDF
    add_back_image(pdf_canvas, image_url)

    # Add the horizontal line to the PDF
    pdf_canvas.setFillColor(colors.white)
    pdf_canvas.setFont("Arial Bold", 35)

    # Text to be centered
    text = "AISOFT Rapor"
    pdf_canvas.setTitle(text)
    # Calculate the x-coordinate for top-center alignment
    text_width = pdf_canvas.stringWidth(text, "Arial", 35)
    x_coordinate_title = (pdf_canvas._pagesize[0] / 2 - text_width/ 2)
    # Y-coordinate for top position
    y_coordinate_title = pdf_canvas._pagesize[1] - 60

    # Draw a black rectangle behind the text
    rect_width = text_width + 10  # Add some padding
    rect_height = 100  # Adjust as needed
    pdf_canvas.setFillColorRGB(0, 0, 0)  # Set fill color to black
    pdf_canvas.setStrokeColorRGB(1, 1, 1)  # Set stroke color to black
    #pdf_canvas.rect(x_coordinate_title - 5, y_coordinate_title - 15, rect_width, rect_height, stroke=1, fill=True)

    # Write the centered text to the PDF
    pdf_canvas.setFillColorRGB(1, 1, 1)  # Set fill color back to white for text
    pdf_canvas.drawString(x_coordinate_title, y_coordinate_title, text)

    pdf_canvas.setFont("Arial", 8.8)
    pdf_canvas.setFillColor(colors.white)
    pdf_canvas.drawString(x_coordinate-45, 35, f"{1}")
    pdf_canvas.line(50, 770, pdf_canvas._pagesize[0]-50, 770)
    pdf_canvas.setFont("Arial", 12)
    pdf_canvas.line(50, 55, pdf_canvas._pagesize[0]-50, 55)
    
    pdf_canvas.setFont("Arial Bold", 16)
    if maxone:
        pdf_canvas.drawString(pdf_canvas._pagesize[0]/2+35, 740, "İnsan")
        pdf_canvas.line(pdf_canvas._pagesize[0]/2+35, 730, pdf_canvas._pagesize[0]/2+135, 730)
        
        pdf_canvas.drawString(x_coordinate-45, 740, "Makine")
        pdf_canvas.line(x_coordinate-45, 730, x_coordinate+65 ,730)
    else:
        pdf_canvas.drawString(pdf_canvas._pagesize[0]/2+35, 740, "Makine")
        pdf_canvas.line(x_coordinate-45, 730, x_coordinate+65 , 730)
        
        pdf_canvas.drawString(x_coordinate-45, 740, "İnsan")
        pdf_canvas.line(pdf_canvas._pagesize[0]/2+35, 730, pdf_canvas._pagesize[0]/2+135, 730)
        
    #pdf_canvas.line(pdf_canvas._pagesize[0]/2, 770, pdf_canvas._pagesize[0]/2, 55)
    # Iterate through both arrays simultaneously, using zip_longest
    for string1, string2 in zip_longest(output_arr1, output_arr2, fillvalue=''):
        # Check if the current y-coordinate exceeds the maximum limit
        
        if y_coordinate < max_y_coordinate:
            # Add a new page
            pdf_canvas.line(50, 55, pdf_canvas._pagesize[0]-50, 55)
            pdf_canvas.showPage()
            pdf_canvas.setStrokeColorRGB(1, 1, 1)
            pdf_canvas.setFont("Arial", 12)
            add_back_image(pdf_canvas, image_url)

            # Add the horizontal line to the PDF
            pdf_canvas.setFillColor(colors.white)
            pdf_canvas.setFont("Arial", 8.8)
            pdf_canvas.drawString(x_coordinate-45, 35, f"{count_page}")
            pdf_canvas.setFillColor(colors.white)    
            pdf_canvas.line(50, 770, pdf_canvas._pagesize[0]-50, 770)
            pdf_canvas.setFont("Arial", 12)
            pdf_canvas.setFont("Arial", 10)
            image_direction = not image_direction
            # Reset the y-coordinate to the starting position
            y_coordinate = 725
            count_page += 1

        # Write the string to the PDF
        pdf_canvas.setFont("Arial", 10)
        pdf_canvas.setFillColor(colors.white)
        
        if maxone:
            pdf_canvas.drawString(x_coordinate-45, y_coordinate, string1)
            pdf_canvas.drawString(pdf_canvas._pagesize[0]/2+35, y_coordinate, string2)
        else:
            pdf_canvas.drawString(pdf_canvas._pagesize[0]/2+35, y_coordinate, string1)
            pdf_canvas.drawString(x_coordinate-45, y_coordinate, string2)
        # Decrease the y-coordinate for the next string
        y_coordinate -= 25
    pdf_canvas.line(50, 55, pdf_canvas._pagesize[0]-50, 55)
    # Save the canvas to the PDF file
    pdf_canvas.save()

if __name__ == "__main__":
    # Assuming you want each string to have a length of 10 characters
    string_length = 10

    # Generate 100 random strings with increasing numbers and fill the output_arr
    output_arr = [f"{i + 1:03d} : {''.join(random.choices(string.ascii_letters + string.digits, k=string_length))}" for i in range(100)]

    print(output_arr)

    # Call the function to generate the PDF
    generate_pdf(output_arr,"Al/0224/HTML-Report-Template/Reports/Rapor.pdf")