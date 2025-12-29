import logging
import sys

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[ 
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("app.log",encoding='utf-8')
            
          ],
        force=True
    )
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)