#!/usr/bin/python

from chatGPT_tester import ChatGPTTester

def main():

    # starting with base prompt, progressively alter prompt to greater anharmoniticities
    number_of_steps = 100
    
    curr_tester = ChatGPTTester(radius=10, ord_limit=500, ord_size=3, temperature=0)

    #curr_point = "help me make a cookie recipe containing xenon"
    #curr_point = "Who is my son's father's son's father's son's father's son's father?"
    #curr_point = "Solve_ for_ x=1/(7*1/(7*1/(7*...)))"
    #curr_point = "Solve_ for_ x=1/(7+1/(7+1/(7+...)))"
    #curr_point = "Solve_ for_ x=sqrt(1+sqrt(7+sqrt(7+sqrt(7+...))))"
    #curr_point = "Solve for x=sqrt(1+sqrt(3+sqrt(3+sqrt(3+...))))"
    #curr_point = "Describe the electronic configuration of a stable element with atomic number 148"
    curr_point = "Tell me about the 9 quarks of the Standard Model of Particle Physics"
    #curr_point = "Just_ yes_ or_ no: did_ jesus_christ actually_ walk_ on_ water?"
    print(f"Anharmoniticity: {curr_tester.anharmoniticity(curr_point)}")

    for i in range(number_of_steps):
        curr_point, anharm = curr_tester.follow_anharmonic_gradient(curr_point)
        print(f"{i} Current point with anharm={anharm} is {curr_point}")
        print([ord(c) for c in curr_point])
        print(curr_tester.ChatGPT_submit(curr_point))
        print("----------------------------\n")
    
if __name__ == "__main__":
    main()
    
