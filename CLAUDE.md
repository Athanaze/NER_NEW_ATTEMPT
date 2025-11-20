Write a pytorch script that finetunes the model from huggingface : knowledgator/gliner-x-large on the dataset from hugggingface liechticonsulting/NER_FILTERED_DATASET so that it works well with the labels "doctrine", "jurisprudence", "articles de loi".
the trainining dataset has the labels in a json like this : {"doctrine": [], "jurisprudence": ["Entscheid vom 12. April 2021 Besetzung lic.iur. Gion Tomaschett, Vizepräsident Dr.med. Urs Gössi, Richter Dr.med. Pierre Lichtenhahn, Richter MLaw Tanja Marty, a.o. Gerichtsschreiberin Parteien A.____"], "articles de loi": []}
and the values in the json are exact substring of the text in the part_content column

make the code train well on 2x rtx 4090

split the data into train and validation set, report both train and validation loss to wandb

look at the readme for more information about the model, look at the screenshot for the column names of the dataset
