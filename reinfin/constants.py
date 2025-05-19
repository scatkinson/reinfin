from collections import OrderedDict

# pathing constants
# -Important-: each path below should be specific to the machine running the code

# absolute path to the data directory where the extracted files will be saved
#  each social media platform has a subdirectory here
DATA_DIRECTORY = (
    "/Users/scott.atkinson/Library/CloudStorage/Box-Box/TO47_Seaside_Social_Media/Data/"
)
# absolute path to the working directory containing this code base
WORKING_DIRECTORY = "/Users/scott.atkinson/src/socialtorch/"
# absolute path to a temporary directory that is used for testing purposes
TMP_DIRECTORY = "/tmp/"

# tumblr constants
TYPE_COL = "type"
BLOG_NAME_COL = "blog_name"
BLOG_DESCRIPTION_COL = "blog_description"
ID_STRING_COL = "id_string"
POST_URL_COL = "post_url"
DATE_COL = "date"
STATE_COL = "state"
FORMAT_COL = "format"
TAGS_COL = "tags"
SHORT_URL_COL = "short_url"
SUMMARY_COL = "summary"
CAPTION_COL = "caption"
BODY_COL = "body"

# youtube constants
TEXT_ORIGINAL_COL = "textOriginal"
AUTHOR_DISPLAY_NAME_COL = "authorDisplayName"
PUBLISHED_AT_COL = "publishedAt"
YOUTUBE_TIME_FORMAT_STR = "%Y-%m-%dT%H:%M:%SZ"
KIND_COL = "kind"
CATEGORY_ID_COL = "videoCategoryId"
CATEGORY_NAME_COL = "videoCategoryName"
VIDEO_TITLE_COL = "videoTitle"
VIDEO_DESCRIPTION_COL = "videoDescription"

# reddit constants
AUT_NAME_COL = "aut_name"
CREATED_UTC_COL = "created_utc"
SUBREDDIT_NAME_PREFIXED = "subreddit_name_prefixed"
BANNED_BY_COL = "banned_by"
IS_SUBMITTER_COL = "is_submitter"
LINK_TITLE_COL = "link_title"
LINK_ID_COL = "link_id"
DEPTH_COL = "depth"
LINK_FLAIR_COL = "link_flair"
LINK_FLAIR_TEXT_COL = "link_flair_text"
AUTHOR_FLAIR_TEXT_COL = "author_flair_text"
SUBMISSION_AUTHOR_FLAIR_TEXT_COL = "submission_author_flair_text"
SUBMISSION_CREATED_UTC_COL = "submission_created_utc"
SUBMISSION_AUTHOR_COL = "submission_author"
SUBMISSION_ID_COL = "submission_id"
SUBMISSION_BANNED_BY_COL = "submission_banned_by"
SELFTEXT_COL = "selftext"
COMMENT_ID_COL = "comment_id"
DATA_TYPE_COMMENTS = "comments"
DATA_TYPE_SUBMISSIONS = "submissions"
GET_BY_AUTHOR = "author"
GET_BY_SUBREDDIT = "subreddit"
GET_BY_SUBMISSION_ID = "submission_id"
METHOD_CONTROVERSIAL = "controversial"
METHOD_GILDED = "gilded"
METHOD_HOT = "hot"
METHOD_NEW = "new"
METHOD_RISING = "rising"
METHOD_TOP = "top"
UNIT_USER = "user"
UNIT_SUBREDDIT = "subreddit"

COMMENT_METHODS = [
    METHOD_CONTROVERSIAL,
    METHOD_GILDED,
    METHOD_HOT,
    METHOD_NEW,
    METHOD_RISING,
    METHOD_TOP,
]
TIME_FILTER_METHODS = [METHOD_CONTROVERSIAL, METHOD_GILDED, METHOD_RISING, METHOD_TOP]

# flickr constants
AUTHOR_COL = "author"
AUTHOR_NAME_COL = "authorname"
DATECREATE_COL = "datecreate"
PATH_ALIAS_COL = "path_alias"
REAL_NAME_COL = "realname"
TITLE_COL = "title"
DESCRIPTION_COL = "description"
PHOTO_COMMENT_COUNT_COL = "photo_comment_count"
PHOTO_ID_COL = "photo_id"
CONTENT_COL = "_content"
TAG_KEY = "tag"
PHOTOS_KEY = "photos"
PHOTO_KEY = "photo"
DATEUPLOADED_KEY = "dateuploaded"

# stack exchange constants
CREATION_DATE_COL = "creation_date"
QUESTION_ID_COL = "question_id"
ANSWER_ID_COL = "answer_id"
COMMENT_ID_COL = "comment_id"
POST_ID_COL = "post_id"

# twitter constants
IN_REPLY_TO_USER_ID_COL = "in_reply_to_user_id"
RECENT_TWEET_STR = "recent_tweet"
CREATED_AT_KEY = "created_at"
AUTHOR_ID_KEY = "author_id"
USER_FETCHED_COL = "user_fetched"

# uniform output fields
CONSTRUCT_COL = "construct"
SUBCONSTRUCT_COL = "subconstruct"
ASSESSMENT_COL = "assessment"
RECORD_ID_COL = "record_id"
ID_COL = "id"
USER_ID_COL = "user_id"
USERNAME_COL = "username"
NAME_COL = "name"
USER_INFO_COL = "user_info"
VERIFIED_COL = "verified"
TEXT_COL = "text"
TIMESTAMP_COL = "timestamp"
LANGUAGE_COL = "language"
LABEL_COL = "label"
SOURCE_COL = "source"
IS_REPLY_COL = "is_reply"
BANNED_COL = "banned"
SUBJECT_COL = "subject"
SUBJECT_DESCRIPTION_COL = "subject_description"
RELEVANCE_COL = "relevance"

# cleaning columns
TEXT_CLEANED_COL = "text_cleaned"
CLEANED_CHCECKPOINT1_COL = "text_cleaned1"
CLEANED_CHCECKPOINT2_COL = "text_cleaned2"
COUNT_BY_USER_COL = "count_data_from_user"
USERTAGS_COL = "usertags"
USERTAGS_COUNT_COL = "usertag_count"
HASHTAGS_COL = "hashtags"
HASHTAGS_COUNT_COL = "hashtag_count"
EMOTICON_LIST_COL = "emoticon_list"
EMOTICON_MEANING_COL = "emoticon_meaning"
UNIEMOJI_LIST_COL = "unicode_emojilist"
EMOJI_LIST_COL = "emojilist"
ALL_EMOJI_LIST_COL = "all_emojilist"
ALL_EMOJI_MEANING_COL = "all_emojilist_text"
ALL_EMOJI_COUNT_COL = "emoji_count"
FT_LANGUAGE_COL = "iso_lang_1"
FT_PROB_COL = "prob_1"
REDDIT_QUOTES_COL = "quoted_material"

# pandas datatype strings
OBJECT = "object"
INT64 = "int64"
FLOAT64 = "float64"
BOOL = "bool"
DATETIME64 = "datetime64"
CATEGORY = "category"

EXTRACT_COLS = OrderedDict(
    {
        RECORD_ID_COL: OBJECT,
        USER_ID_COL: OBJECT,
        USERNAME_COL: OBJECT,
        USER_INFO_COL: OBJECT,
        VERIFIED_COL: BOOL,
        TEXT_COL: OBJECT,
        TIMESTAMP_COL: INT64,
        LANGUAGE_COL: OBJECT,
        LABEL_COL: OBJECT,
        TAGS_COL: OBJECT,
        SOURCE_COL: OBJECT,
        IS_REPLY_COL: INT64,
        BANNED_COL: BOOL,
        SUBJECT_COL: OBJECT,
        SUBJECT_DESCRIPTION_COL: OBJECT,
        CONSTRUCT_COL: OBJECT,
        SUBCONSTRUCT_COL: OBJECT,
        ASSESSMENT_COL: OBJECT,
        RELEVANCE_COL: OBJECT,
    }
)

FLICKR_STR = "Flickr"
TUMBLR_STR = "Tumblr"
YOUTUBE_STR = "YouTube"
REDDIT_STR = "Reddit"
STACKEXCHANGE_STR = "StackExchange"
TWITTER_STR = "Twitter"

DATE_FORMAT_STR = "%Y/%m/%d"


TUMBLR_RESPONSE_FIELDS = [
    TYPE_COL,
    BLOG_NAME_COL,
    ID_STRING_COL,
    POST_URL_COL,
    DATE_COL,
    TIMESTAMP_COL,
    STATE_COL,
    FORMAT_COL,
    TAGS_COL,
    SHORT_URL_COL,
    SUMMARY_COL,
]

FLICKR_RESPONSE_FIELDS = [
    ID_COL,
    AUTHOR_COL,
    DATECREATE_COL,
    TITLE_COL,
    DESCRIPTION_COL,
    PHOTO_COMMENT_COUNT_COL,
    PHOTO_ID_COL,
    TAGS_COL,
]

# Master Extract Constants
SUBREDDIT_COL = "subreddit"
REDDIT_SUBMISSION_COL = "reddit_submission"
REDDIT_AUTHOR_COL = "reddit_author"
TWITTER_USER_COL = "twitter_user"
TWITTER_TAG_COL = "twitter_tag"
YOUTUBE_VIDEO_COL = "youtube_video"
YOUTUBE_PLAYLIST_COL = "youtube_playlist"
YOUTUBE_CHANNEL_COL = "youtube_channel"
TUMBLR_TAG_COL = "tumblr_tag"
TUMBLR_USER_COL = "tumblr_user"
STACKEXCHANGE_SITE_COL = "stackexchange_site"
STACKEXCHANGE_USER_COL = "stackexchange_user"
FILEPATH_COL = "filepath"
USER_SOURCE_COL = "user_source"
RECORD_COUNT_COL = "record_count"
LINK_COL = "link"

LOGGING_PATH_KEY = "logging_path"
PIPELINE_ID_KEY = "pipeline_id"
KEYS_PATH_KEY = "keys_path"
STARTING_POS_KEY = "starting_pos"
LIMIT_KEY = "limit"
AUTHORS_KEY = "authors"
SUBREDDITS_KEY = "subreddits"
SUBMISSION_IDS_KEY = "submission_ids"
VIDEO_IDS_KEY = "video_ids"
AUTHOR_IDS_KEY = "author_ids"
CHANNEL_IDS_KEY = "channel_ids"
PLAYLIST_IDS_KEY = "playlist_ids"
CHANNEL_INFO_KEY = "channel_info"
TAG_DICT_KEY = "tag_dict"
QUANTITY_KEY = "quantity"
SITE_KEY = "site"
BEFORE_KEY = "before"
CONSTRUCT_KEY = "construct"
SUBCONSTRUCT_KEY = "subconstruct"
ASSESSMENT_KEY = "assessment"
RELEVANCE_KEY = "relevance"
SUBMISSION_FILEPATH_COL = "submission_filepath"
FLAIR_FILEPATH_COL = "flair_filepath"
PLATFORM_COL = "platform"
LABEL_KEY = "label"

MASTER_LOGGING_PATH = "bin/extract/master/logs"

REDDIT_KEYS_PATH = "soctor/extract/reddit/keys/keys1.yml"
YOUTUBE_KEYS_PATH = "soctor/extract/youtube/keys/keys.txt"
YOUTUBE_STARTING_POS = 0
TUMBLR_KEYS_PATH = "soctor/extract/tumblr/keys/keys.yml"
STACKEXCHANGE_KEYS_PATH = "soctor/extract/stackexchange/keys/keys.yml"

TUMBLR_TAG_QUANTITY = 2500
STACKEXCHANGE_LIMIT = 10000
YOUTUBE_CUTOFF = 200 * 1000

TWITTER_USER_REPLACE_LOOP_CUTOFF = 3
TWITTER_USER_DIFFERENCE_THRESHOLD = 10

TWEET_FIELD_DICT = {
    CREATED_AT_KEY: TIMESTAMP_COL,
    ID_COL: RECORD_ID_COL,
    AUTHOR_ID_KEY: USER_ID_COL,
    TEXT_COL: TEXT_COL,
    IN_REPLY_TO_USER_ID_COL: IN_REPLY_TO_USER_ID_COL,
    LANGUAGE_COL: LANGUAGE_COL,
}

# cleaning report constants
INITIAL_COUNT_COL = "initial_count"
FINAL_COUNT_COL = "final_count"
TOTAL_DROPPED_COUNT_COL = "total_dropped_count"
DROPPED_DUPLICATES_COUNT_COL = "dropped_duplicates_count"
DROPPED_VERIFIED_OR_BOTS_COUNT_COL = "dropped_verified_or_bots_count"
DROPPED_FOREIGN_CHARACTERS_COUNT_COL = "dropped_foreign_characters_count"
DROPPED_URLS_COUNT_COL = "dropped_urls_count"
DROPPED_SHORT_POSTS_COUNT_COL = "dropped_short_posts_count"
DROPPED_NONENGLISH_COUNT_COL = "dropped_nonenglish_count"
DROPPED_ONLY_EMOJI_COUNT_COL = "dropped_only_emoji_count"
DROPPED_NONLATIN_COUNT_COL = "dropped_nonlatin_count"
TIME_TO_CLEAN_COL = "time_to_clean"
TOTAL_DROPPED_PCT_COL = "total_dropped_pct"

GENDER_COL = "gender"
AGE_COL = "age"
AGE_RANGE_COL = "age_range"
AGE_MIN_COL = "age_min"
AGE_MAX_COL = "age_max"
BIRTHDAY_COL = "birthday"
AGE_FLAG_COL = "age_flag"
GENDER_FLAG_COL = "gender_flag"
UNDER_25_COL = "under_25"
HAS_EMOJI_COL = "has_emoji"
X_STR = "X"
M_STR = "M"
F_STR = "F"
NB_STR = "NB"
NONE_STR = "None"

PATTERN_KEY = "pattern"
MIN_KEY = "min"
MAX_KEY = "max"
AGERANGEPATTERNS = [
    {
        PATTERN_KEY: r"i am a (HS|high school) freshman|i am a freshman in (HS|high school)",
        MIN_KEY: 13,
        MAX_KEY: 16,
    },
    {
        PATTERN_KEY: r"i am a (HS|high school) sophomore|i am a sophomore in (HS|high school)",
        MIN_KEY: 14,
        MAX_KEY: 17,
    },
    {
        PATTERN_KEY: r"i am a (HS|high school) junior|i am a junior in (HS|high school)",
        MIN_KEY: 15,
        MAX_KEY: 18,
    },
    {
        PATTERN_KEY: r"i am a (HS|high school) senior|i am a senior in (HS|high school)",
        MIN_KEY: 16,
        MAX_KEY: 19,
    },
    {
        PATTERN_KEY: r"i am a college freshman|i am a freshman in college",
        MIN_KEY: 17,
        MAX_KEY: 21,
    },
    {
        PATTERN_KEY: r"i am a college sophomore|i am a sophomore in college",
        MIN_KEY: 18,
        MAX_KEY: 22,
    },
    {
        PATTERN_KEY: r"i am a college junior|i am a junior in college",
        MIN_KEY: 19,
        MAX_KEY: 24,
    },
    {
        PATTERN_KEY: r"i am a college senior|i am a senior in college",
        MIN_KEY: 20,
        MAX_KEY: 25,
    },
    {PATTERN_KEY: r"i am retired", MIN_KEY: 40, MAX_KEY: 100},
    {PATTERN_KEY: r"i am middle[ \-]aged", MIN_KEY: 40, MAX_KEY: 65},
    {PATTERN_KEY: r"i am a grandparent", MIN_KEY: 40, MAX_KEY: 100},
]
X0SPATTERNS = [
    {PATTERN_KEY: r"i am in my early (\d)0\'?s", MIN_KEY: "0", MAX_KEY: "4"},
    {PATTERN_KEY: r"i am in my mid (\d)0\'?s", MIN_KEY: "2", MAX_KEY: "8"},
    {PATTERN_KEY: r"i am in my late (\d)0\'?s", MIN_KEY: "7", MAX_KEY: "9"},
    {PATTERN_KEY: r"i am in my (\d)0\'?s", MIN_KEY: "0", MAX_KEY: "9"},
]

TURNING_PATTERN = r"(?<!\")(?:i am turning (\d\d)|i am going to be (\d\d)(?!\d%))(?!\")"
X_YRS_OLD_PATTERN = r"(?<!\")(?:i am (\d\d) years old|i just turned (\d\d)|i am a (\d\d) year old)(?!\")|it is my (\d\d)(?:th|st|rd|nd) birthday"
MALE_PATTERN = r"(?<!\")(?:i am a man($|\W)|i am (a|the) (father|dad|husband|son|brother)($|\W))(?!\")"
FEMALE_PATTERN = r"(?<!\")(?:i am a woman($|\W)|i am (a|the) (mother|mom|wife|daughter|sister)($|\W))(?!\")"
NONBINARY_PATTERN = r"(?<!\")(?:i am non\-?binary)(?!\")"

AGE_GENDER_PATTERN = r"(?<!\"|„)(i|my|me|^) \((\d\d)[\s\/\|]?([mf])\)(?!\"|“)"

FLAIR_PATTERN = r"^([1-6]\d)[\s\/\|]?([MF])($|\s)"
FEMALE_FLAIR_PATTERN = r"(\b|^)(female|woman|gal|mother|mom|wife|sister)(\b|$)"
MALE_FLAIR_PATTERN = r"(\b|^)(male|man|guy|father|dad|husband|brother)(\b|$)"

# Model constants
ACCURACY_KEY = "accuracy"
AUC_KEY = "auc"
LOSS_KEY = "loss"
EPOCHS_KEY = "epochs"
LEARNING_RATE_KEY = "learning_rate"
PATIENCE_KEY = "patience"
FACTOR_KEY = "factor"
PLATEAU_VAL_KEY = "plateau_val"
BATCH_SIZE_KEY = "batch_size"
MAX_TOKENCOUNT_KEY = "max_tokencount"
MODEL_TYPE_KEY = "model_type"
PLOT_SAVE_PATH_KEY = "plot_save_path"
ROC_SAVE_PATH_KEY = "roc_save_path"
LOGFILE_KEY = "logfile"
TRAIN_KEY = "train"
VAL_KEY = "val"
TEST_KEY = "test"
PROBA_COL = "proba"

# Analysis constants
POP_AGE_SCORES_COL = "pop_age_scores"
POP_GENDER_SCORES_COL = "pop_gender_scores"

AGE_SCORE_COL = "age_score"
GENDER_SCORE_COL = "gender_score"
MEAN_COL = "mean"
MEDIAN_COL = "median"
STD_COL = "std"
NUNIQUE_COL = "nunique"

AGE_SCORE_MEAN_COL = "age_score_mean"
AGE_SCORE_MEDIAN_COL = "age_score_median"
AGE_SCORE_STD_COL = "age_score_std"
GENDER_SCORE_MEAN_COL = "gender_score_mean"
GENDER_SCORE_MEDIAN_COL = "gender_score_median"
GENDER_SCORE_STD_COL = "gender_score_std"

AGE_SCORE_COHENS_D_COL = "age_score_cohens_d"
GENDER_SCORE_COHENS_D_COL = "gender_score_cohens_d"

WEIGHT_COL = "weight"
