try: 
    from logger import logger 

except ImportError as e:
    print("error importing module in model evaluation")

class ModelEvaluation():
    def initiate_model_evaluation(self):
        try:
            pass
        except Exception as e:
            logger.info(f"----------error in model evaluation".ljust(60, '-'))