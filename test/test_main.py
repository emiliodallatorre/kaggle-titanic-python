import os.path
from unittest import TestCase
import main


class TestMain(TestCase):
    def test_main(self):
        os.chdir("..")

        if os.path.exists("output.csv"):
            os.remove("output.csv")

        main.main()

        if not os.path.exists("output.csv"):
            self.fail()
