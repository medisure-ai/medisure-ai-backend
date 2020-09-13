from summarizer4u import summary

letterpath= "../resources/denial_letter.txt"

def read_and_summarize(letterpath):
	f= open("../resources/denial_letter.txt","r")
	letter=f.read() 

	result= summary(letter)
	print(result)
	return result




read_and_summarize(letterpath)