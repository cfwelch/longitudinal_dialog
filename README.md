

# Learning from personal longitudinal dialog data
This is a project to allow people to convert their Google Hangouts, Facebook Messenger, iMessage, and SMS Backup (Android Application) messages into a single format and explore this data.


## Prerequisites
The code must be run with Python3 (tested with 3.6.4). Install the following Python packages: 

```zsh
pip install msgpack nltk termcolor beautifulsoup4 numpy prettytable python-dateutil tqdm pandas seaborn visdom
```

You may also have to download packages within NLTK. To generate dot files and LaTeX reports you need to install:

```zsh
# Ubuntu
sudo apt-get install latexmk texlive texlive-latex-extra graphviz
# Mac
brew install latexmk texlive texlive-latex-extra graphviz
```

If you want to run prediction experiments you must install pytorch. This is not necessary if you are just generating and looking at statistics. To install pytorch go to [the pytorch website](http://pytorch.org/) and select your OS, package manager, Python version and CUDA version to get the install command(s).

Also, if you want to run experiments you need to scikit and download the [Common Crawl GloVe embeddings](http://nlp.stanford.edu/data/glove.840B.300d.zip) and change your EMBEDDING_LOCATION in the utils.py file to point to where you have them downloaded.

## To download your data
You can download Google Hangouts data from [their takeout page](https://takeout.google.com/settings/takeout) and Facebook data if you follow [their download instructions](https://www.facebook.com/help/302796099745838). Both services will take some time to generate an archive of your data. Your iMessage data is probably stored in ~/Library/Messages/chat.db.

I have noticed that Google Hangouts sometimes does not download all of your messages if you select other things to download at the same time. I click the 'Select None' button and then click the Google Hangouts button to make sure that is the only one I am downloading.


## Generating Statistics

```python
f_my_name, p_fix = ['Charlie Welch', 'charlie']
GOOGLE_PATH = ''
FACEBOOK_PATH = ''
IMESSAGE_PATH = ''
```

1. Edit the preceding lines in utils.py

    Edit f_my_name to the name you use with your social media accounts. p_fix is used for directory names so you can just change that to 'firstname' (Note: This exists in case you want to separate your workspace to work with multiple peoples data. In order to do this you just need to change the names whenever you switch). The Facebook, iMessage, and Google paths should be changed to wherever your data is.

    Note: There is a variable named DATA_DIR. This is the name of a directory that will automatically be created and where generated data files will be placed. It is not the location of your downloaded data.

2. Run user_annotator.py

    This script will loop over your Google, iMessage, and Facebook files and ask you if the names of people are correct. If you talk to someone on both platforms you will want to change the name so that it combines the files when it is doing the analysis. The script will then ask you questions about each person in your dataset. These are questions that I thought would be interesting, which split people into different groups like family, co-workers, classmates, etc.

    When you are naming people you can view random samples of conversation with that person to try and figure out who it is if you aren't sure. If it ends up being difficult, you may have to search for the name or conversation on the appropriate platform to figure out who it is. If the conversation is with an automated service or is spam you can filter these messages so that the scripts will ignore them. Your options are presented at each step.

    The user annotator saves a JSON file in the people folder at each step. If the script fails or is closed you can resume where you left off. If you make a mistake you can edit the JSON file directly.

3. Run the message_formatter.py script to convert your data into a common format using the names from the user_annotator.

4. Run the data merger with 'python people_merger.py'. This combines multiple files from the same person to make processing faster.

5. Run the augmentor with 'python augmentor.py -all'. This will add LIWC category, frequency, response time, style matching, and stopword counts to all data files.

6. Run 'python get_stats.py -all' which will create spreadsheets in the stats folder and print some output. You can change the START_TIME variable in utils.py to change the time range of messages to look at. It will include all messages sent from this time until the current time. There are separate flags if you do not want to generate all of the information but you will need all of it to generate LaTeX reports.

7. If you have run step 5 with the 'all' flag you can generate a PDF with figures and tables related to your data. Run 'python gen_latex.py' to create this file in the latex subfolder.


## Data format
The code does not currently support formats other than Google Hangouts and Facebook, but if a converter is written the other scripts will read all of the files from the data folder. The converted files should be separated by underscores and the first part should be the format name.

The common message file format is a binary msgpack object which contains:
```python
conversation[b'with'] = 'First Last'     # <type 'str'>
conversation[b'messages'] = message_list # <type 'list'>
```

The message list object contains:
```python
message_list[i][b'user'] = 'Speaker Name'            # <type 'str'>
message_list[i][b'date'] = 'ISO 8601 formatted date' # <type 'str'>
message_list[i][b'text'] = 'message text'            # <type 'str'>
```


## Augmented format
After running the augmentor you will have additional useful fields attached to messages. These are used by other scripts but you can use them for whatever you want:
```python
message_list[i][b'liwc_counts'] = [cat1_count, cat2_count] # <type 'list'> of <type 'int'>
message_list[i][b'liwc_words'] = [['liwc_cat1_word1', 'liwc_cat1_word2'], ['liwc_cat2_word1', 'liwc_cat2_word2']] # <type 'list'> of <type 'list'> of <type 'str'>
message_list[i][b'words'] = num_words # <type 'int'>

message_list[i][b'stopword_count'] = num_words # <type 'int'>
message_list[i][b'turn_change'] = tc # <type 'bool'>
message_list[i][b'response_time'] = num_seconds # <type 'int'>

message_list[i][b'lsm'] = style_matching_score # <type 'int'>

message_list[i][b'all_freq'] = all_time_count # <type 'int'>
message_list[i][b'month_freq'] = month_count # <type 'int'>
message_list[i][b'week_freq'] = week_count # <type 'int'>
message_list[i][b'day_freq'] = day_count # <type 'int'>
```


## Generated Statistics
If you run get_stats.py it will create a lot of files in your stats folder. This is what each file contains:

 * day_of_week_stats.csv -- Breakdown of groups and how much you converse with them each by the day of the week.
 * hour_of_day_stats.csv -- Breakdown of groups and how much you converse with them each hour of the day.
 * liwc_grouped.csv -- Incoming/outgoing normalized counts of each LIWC category with each group of people aggregated over time.
 * liwc_person.csv -- Counts of each LIWC category in conversations with each person.
 * liwc_person_normalized.csv -- Normalized counts of each LIWC category in conversations with each person.
 * month_season_dists.txt -- Number of messages exchanged in each month and each season.
 * msgs_aggregate.csv -- Number of people spoken to and messages exchanged per each month bin.
 * msgs_over_time_by_person.csv -- Number of messages exchanged with each person per each month bin.
 * response_time_bins.txt -- Bins of your response time to other people by seconds. The left number is seconds and right number is number of responses that fall in that bin.
 * total_messages_per_category.csv -- For each group shows the incoming and outgoing number of words (I(W) and O(W)) and messages (I(M) and O(M)) as well as the totals (T(W) and T(M)).
 * total_messages_per_person.csv -- Incoming and outgoing messages and words for each person you speak to as well as the ratios (Out ratio of 10, for instance would mean that you use an average of 10 words per message when speaking to this person).
 * total_uniques.txt -- Total unique tokens and messages by you or others.
 * freq_by_stat/group_messages_per_month.csv -- Number of messages exchanged with each group over each month bin.
 * freq_by_stat/group_resp_time_per_month.csv -- Your average response time to each group per each month bin.
 * freq_by_stat/group_words_per_month.csv -- Number of words exchanged with each group over each month bin.
 * freq_by_stat/group_wpm_per_month.csv -- Average words per message exchanged with each group over each month bin.
 * freq_by_stat/messages_per_month.csv -- Number of messages exchanged with each person over each month bin.
 * freq_by_stat/words_per_month.csv -- Number of words exchanged with each person over each month bin.
 * freq_by_stat/wpm_per_month.csv -- Average words per message exchanged with each person over each month bin.
 * liwc_by_category/* -- Spreadsheets for each LIWC category that show normalized counts for that category for each person per month bin.
 * liwc_by_group/* -- Spreadsheets for each LIWC category that show normalized counts for that category for each group per month bin.
 * liwc_by_person/* -- Spreadsheets for each person that show normalized counts for each LIWC category per month bin. This includes a spreadsheet for yourself so that you can look at how your way of speaking changes over time.


## Adding user attributes

You can add user attributes by editing the user_annotator.py script. There are two dictionaries named tag_set and tag_description listed at the top of the file. Edit tag_set to have the attribute name and a list of options for values of that attribute and edit the tag_description to contain instructions for the user. After you've done this, if you rerun the annotator it will ask you to annotate people who do not already have this attribute annotated.
