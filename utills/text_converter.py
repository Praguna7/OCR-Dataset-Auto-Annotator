import subprocess

class TextConverter:

    @staticmethod
    def convert_text_with_js(input_text):
        # Call the Node.js process
        result = subprocess.run(
            ['node', 'utills/convert.js', input_text],
            capture_output=True,
            text=True,
            encoding='utf-8'  # Specify UTF-8 encoding
        )

        # Check for errors
        if result.returncode != 0:
            print("Error:", result.stderr)
            return None

        # Return the output from the JavaScript function
        return result.stdout.strip()
