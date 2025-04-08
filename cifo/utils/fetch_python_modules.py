import requests
from bs4 import BeautifulSoup
from datetime import datetime

def get_modules():
    url = 'https://docs.python.org/3/py-modindex.html'

    response = requests.get(url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'lxml')
        
        # Find the first table with class 'indextable modindextable'
        table = soup.find('table', class_='indextable modindextable')

        if table:
            # Find all <a> tags with class 'xref' inside the tbody
            module_names = []
            for code_tag in table.find_all('code', class_='xref'):
                # The class attribute contains the module name
                module_name = code_tag.get_text()
                if module_name:
                   module_names.append(module_name.strip())
            
            # If we found any module names, save them to a file
            if module_names:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"sys_modules_{timestamp}.txt"
                with open(filename, 'w') as file:
                    for module in module_names:
                        file.write(module + '\n')
                print(f"Module names saved to {filename}")
            else:
                print("No module names found.")
        else:
            print("No tbody found in the table.")
    else:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")

if __name__ == "__main__":
    get_modules()
