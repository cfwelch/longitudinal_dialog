

from argparse import ArgumentParser

from utils import MSG_TYPE_PATHS
from google_format import gh_format
from facebook_format import fb_format
from imessage_format import imsg_format
from instagram_format import ig_format

def main():
    # parser = ArgumentParser()
    # parser.add_argument("-hs", "--high-sierra", dest="high_sierra", help="If your chat.db was generated from using macOS High Sierra (or later?) you must set this to true because iMessage dates will be stored in a different format. Defaults to False.", default=False, action="store_true")
    # opt = parser.parse_args()

    print('Formatting messages from Google Hangouts...')
    for i in range(len(MSG_TYPE_PATHS['GH'])):
        gh_format(MSG_TYPE_PATHS['GH'][i], i)

    print('Formatting messages from Facebook...')
    for i in range(len(MSG_TYPE_PATHS['FB'])):
        fb_format(MSG_TYPE_PATHS['FB'][i], i)

    print('Formatting messages from iMessage...')
    for i in range(len(MSG_TYPE_PATHS['IMSG'])):
        imsg_format(MSG_TYPE_PATHS['IMSG'][i], i, True) #opt.high_sierra

    print('Formatting messages from Instagram...')
    for i in range(len(MSG_TYPE_PATHS['IG'])):
        ig_format(MSG_TYPE_PATHS['IG'][i], i)

if __name__ == "__main__":
    main()
