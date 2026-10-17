import os
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.opc.constants import RELATIONSHIP_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


def add_docx_hyperlink(paragraph, url, text, color="1155CC", underline=True):
    """Inserts an active clickable hyperlink into a python-docx paragraph."""
    part = paragraph.part
    r_id = part.relate_to(url, RELATIONSHIP_TYPE.HYPERLINK, is_external=True)
    hyperlink = OxmlElement('w:hyperlink')
    hyperlink.set(qn('r:id'), r_id)
    new_run = OxmlElement('w:r')
    rPr = OxmlElement('w:rPr')
    if color:
        c = OxmlElement('w:color')
        c.set(qn('w:val'), color)
        rPr.append(c)
    if underline:
        u = OxmlElement('w:u')
        u.set(qn('w:val'), 'single')
        rPr.append(u)
    new_run.append(rPr)
    new_run.text = text
    hyperlink.append(new_run)
    paragraph._p.append(hyperlink)
    return hyperlink


def build_submission_docx(
    student_name="Saksham Gupta",
    reg_num="230911186",
    repo_url="https://github.com/Saksham-Gupta-GH/fgai-autoregressive-text-generation-timemachine",
    output_filename="SakshamGupta_230911186_FGAI_Assignment2.docx"
):
    doc = Document()
    
    # Margins
    for section in doc.sections:
        section.top_margin = Inches(1.1)
        section.bottom_margin = Inches(1.1)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
    style = doc.styles['Normal']
    style.font.name = 'Times New Roman'
    style.font.size = Pt(11)

    # Institutional Header Logo
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    image_path = os.path.join(repo_root, "images.jpeg")
    if os.path.exists(image_path):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_after = Pt(24)
        run_img = p_img.add_run()
        run_img.add_picture(image_path, width=Inches(5.6))
        
    # Title
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(10)
    run_title = title.add_run("Foundations of Generative AI (ICT4403)")
    run_title.font.name = 'Times New Roman'
    run_title.font.size = Pt(19)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(0, 0, 0)
    
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.paragraph_format.space_after = Pt(30)
    run_sub = sub.add_run("Assignment 2")
    run_sub.font.name = 'Times New Roman'
    run_sub.font.size = Pt(14)
    run_sub.font.bold = True
    run_sub.font.color.rgb = RGBColor(0, 0, 0)
    
    # Metadata Table with clean Table Grid borders
    table = doc.add_table(rows=7, cols=2)
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    meta_rows = [
        ("Student Name", student_name, None),
        ("Registration Number", reg_num, None),
        ("Assignment Title", "Assignment 2 – Autoregressive Text Generation using Simple RNN, Bidirectional RNN, LSTM and GRU", None),
        ("Title of the Developed System", "Comparative Word-Level Autoregressive Text Generation on H. G. Wells' The Time Machine using Simple RNN, Bidirectional RNN, LSTM, and GRU", None),
        ("Dataset Name", "The Time Machine by H. G. Wells (Project Gutenberg eBook #35)", None),
        ("Dataset Link", "", "https://www.gutenberg.org/files/35/35-0.txt"),
        ("GitHub Repository Link", "", repo_url)
    ]
    
    for i, (k, text_prefix, link_url) in enumerate(meta_rows):
        row = table.rows[i]
        c0 = row.cells[0]
        c1 = row.cells[1]
        c0.width = Inches(2.2)
        c1.width = Inches(4.3)
        
        p0 = c0.paragraphs[0]
        p0.paragraph_format.space_before = Pt(11)
        p0.paragraph_format.space_after = Pt(11)
        p0.paragraph_format.line_spacing = 1.15
        r0 = p0.add_run(k)
        r0.font.bold = True
        r0.font.size = Pt(11)
        
        p1 = c1.paragraphs[0]
        p1.paragraph_format.space_before = Pt(11)
        p1.paragraph_format.space_after = Pt(11)
        p1.paragraph_format.line_spacing = 1.15
        if text_prefix:
            r1 = p1.add_run(text_prefix)
            r1.font.size = Pt(11)
        if link_url:
            add_docx_hyperlink(p1, link_url, link_url)
            
    output_path = os.path.join(os.getcwd(), output_filename)
    doc.save(output_path)
    print(f"Generated Word document at: {output_path}")
    return output_path


if __name__ == "__main__":
    build_submission_docx()
