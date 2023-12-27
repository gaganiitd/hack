import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib

from jinja2 import Environment, FileSystemLoader, select_autoescape


# function to render email body using a template
def render_email_body(**kwargs):
    """Render HTML for email using a template."""
    env = Environment(
        loader=FileSystemLoader("templates"),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("email-template.html")
    return template.render(**kwargs)


# Function to handle sending email
async def send_email(name, to, context):
    fromaddr = os.getenv("EMAIL")
    password = os.getenv("APP_PASSWORD")

    if not fromaddr or not password:
        return False, "Email credentials not found"

    message = MIMEMultipart()
    message["To"] = f"{name} <{to}>"
    message["From"] = fromaddr
    message["Subject"] = "Currency Conversion"

    message.attach(MIMEText(render_email_body(**context), "html"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.ehlo("Gmail")
        server.starttls()
        server.login(fromaddr, password)
        server.sendmail(fromaddr, to, message.as_string())

        server.quit()
    except Exception as e:
        return False, str(e)

    return True, "Email sent"
