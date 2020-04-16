import optparse

class hello():
    def greetings(self):
        print('hello')

    def __call__(self):
        self.greetings()

def saysomething():
    parser = optparse.OptionParser(usage='-v')
    parser.add_option("-v", "--verbose", dest="verbose",
                  action="store_true", default=False)
    (options, args) = parser.parse_args()
    hello()()
    if options.verbose:
        print('Yes')
