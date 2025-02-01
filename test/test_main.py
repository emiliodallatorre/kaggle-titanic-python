import os.path
from unittest import TestCase
import main


class TestMain(TestCase):
    OUTPUT_FILE: str = "output.csv"

    def test_main(self):
        os.chdir("..")

        if os.path.exists(self.OUTPUT_FILE):
            os.remove(self.OUTPUT_FILE)

        main.main()

        if not os.path.exists(self.OUTPUT_FILE):
            self.fail()
