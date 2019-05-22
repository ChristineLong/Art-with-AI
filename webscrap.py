import urllib.request
import urllib.parse
import urllib.error
from bs4 import BeautifulSoup
import ssl
import json


# For ignoring SSL certificate errors
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


#Generate loop for different pages
for i in range(2,10):
    url = "https://www.amazon.com/s?i=instant-video&bbn=2858778011&rh=n%3A2858778011%2Cn%3A2858905011%2Cp_n_theme_browse-bin%3A2650376011&dc&page=" + \
      str(i) + "&qid=1555639982&rnid=2858778011&ref=sr_pg_" + str(i)
    html = urllib.request.urlopen(url, context=ctx).read()
    soup = BeautifulSoup(html, 'html.parser')
    html = soup.prettify('utf-8')
    name_list = []

#Since the movie title cannot be stripped directly from the search page
# (title comes from link ref, and link ref does not always show the exact name)

# This block of code will help extract the movie title from the link
# class="avu-full-width" data-automation-id="title"
for x in range(len(name_list)):
    name_of_product= soup.select('h1')[0].text.strip()
    name_list.append(name_of_product)


# Saving the scraped html file
with open('output_file.html', 'wb') as file:
    file.write(html)
# Saving the scraped data in json format
with open('product.json', 'w') as outfile:
    json.dump(product_json, outfile, indent=4)
print ('----------Extraction of data is complete. Check json file.----------')