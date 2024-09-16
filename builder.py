import pandas as pd
import altair as alt
from collections import defaultdict, Counter
from nltk.corpus import stopwords
import numpy as np
import re

import re


class Recap:
    def __init__(self, fp: str, month: int, year: int):
        self.fp = fp

        self.month = month
        self.year = year
        self.last_month = self.month - 1 if self.month > 1 else 12
        self.last_year = self.year if self.month > 1 else self.year - 1

        self.df = self.initiate_df()
        self.this_month_df = self.select_month(self.month, self.year)
        self.last_month_df = self.select_month(self.last_month, self.last_year)
        self.prior_months_df = self.df[
            (self.df["date"].dt.month < self.month)
            & (self.df["date"].dt.year <= self.year)
        ]

    ### Helper Functions ###

    def initiate_df(self):
        df_raw = pd.read_csv(self.fp)
        if not Recap.validate_df(df_raw):
            raise ValueError("Invalid DataFrame")

        df = df_raw.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = Recap.remove_reactions(df)

        return df

    @staticmethod
    def validate_df(df):
        ...
        return True

    @staticmethod
    def remove_reactions(df):
        reactions = ["le encantó", "le gustó", "le dio risa", "le pareció gracioso"]
        pattern = "|".join(reactions)

        return df[~df["body"].str.contains(pattern, flags=re.IGNORECASE, regex=True)]

    @staticmethod
    def days_in_month(month, year) -> int:
        if pd.Timestamp.today() < pd.Timestamp(f"{year}-{month + 1}-01"):
            end = pd.Timestamp.today().date()
        elif month == 12:
            end = f"{year + 1}-01-1"
        else:
            end = f"{year}-{month + 1}-1"
        return len(pd.date_range(start=f"{year}-{month}-01", end=end, freq="D")[:-1])

    @staticmethod
    def count_word_occurrences(text):
        # Use regular expressions to split the text into words
        words = re.findall(r"\b\w+\b", text.lower())

        # Initialize a default dictionary to store word counts
        word_count = defaultdict(int)

        # Iterate through the words and count their occurrences
        for word in words:
            word_count[word] += 1

        return dict(word_count)

    @staticmethod
    def df_word_occurrences(df, ranks=False):
        aggregated_word_count = defaultdict(int)

        # Apply the function to each row in the 'body' column and aggregate the results
        for text in df["body"]:
            word_count = Recap.count_word_occurrences(text)
            for word, count in word_count.items():
                aggregated_word_count[word] += count

        # Convert the aggregated word count to a regular dictionary
        aggregated_word_count = dict(aggregated_word_count)

        # Convert the dictionary to a DataFrame
        word_count = pd.DataFrame(
            aggregated_word_count.items(), columns=["word", "count"]
        )

        # remove stopwords
        stopwords_new = stopwords.words("english")
        stopwords_new.append("like")
        word_count = word_count[~word_count["word"].isin(stopwords_new)]
        word_count = word_count.set_index("word")
        if not ranks:
            return word_count

        word_count.loc[:, "rank"] = (
            word_count["count"].rank(method="dense", ascending=False).astype(int)
        )
        word_count.loc[:, "order_rank"] = (
            word_count["count"].rank(method="first", ascending=False).astype(int)
        )

        return word_count

    @staticmethod
    def count_emojis(df):
        # Define a regular expression pattern for emojis
        emoji_pattern = re.compile(
            "[\U0001F600-\U0001F64F]|"  # Emoticons
            "[\U0001F300-\U0001F5FF]|"  # Symbols & Pictographs
            "[\U0001F680-\U0001F6FF]|"  # Transport & Map Symbols
            "[\U0001F700-\U0001F77F]|"  # Alchemical Symbols
            "[\U0001F780-\U0001F7FF]|"  # Geometric Shapes Extended
            "[\U0001F800-\U0001F8FF]|"  # Supplemental Arrows-C
            "[\U0001F900-\U0001F9FF]|"  # Supplemental Symbols and Pictographs
            "[\U0001FA00-\U0001FA6F]|"  # Chess Symbols
            # "[\U0001FA70-\U0001FAFF]|"  # Symbols and Pictographs Extended-A
            "[\U00002702-\U000027B0]"  # Dingbats
            # "[\U000024C2-\U0001F251]"   # Enclosed Characters
        )

        # Initialize a Counter to count emoji occurrences
        emoji_counter = Counter()

        # Iterate over each entry in the specified column
        for entry in df["body"]:
            if pd.notna(entry):  # Check if the entry is not NaN
                # Find all emojis in the entry
                emojis = emoji_pattern.findall(entry)
                # Update the counter with the found emojis
                emoji_counter.update(emojis)

        # Convert the Counter to a DataFrame
        emoji_counts = (
            pd.DataFrame(emoji_counter.items(), columns=["emoji", "count"])
            .sort_values(by="count", ascending=False)
            .set_index("emoji")
        )

        return emoji_counts

    @staticmethod
    def combined_count(df, on) -> pd.DataFrame:
        me_words = Recap.df_word_occurrences(df[df["is_from_me"] == 1])
        you_words = Recap.df_word_occurrences(df[df["is_from_me"] == 0])
        total = pd.merge(
            me_words, you_words, on=on, suffixes=("_me", "_you"), how="outer"
        ).fillna(0)

        return total

    def select_month(self, month, year):
        return self.df[
            (self.df["date"].dt.month == month) & (self.df["date"].dt.year == year)
        ]

    ### Calendar Heatmap ###

    def make_calendar_heatmap(self) -> alt.Chart:
        def all_days(month, year) -> pd.DataFrame:
            if month == 12:
                return pd.date_range(
                    start=f"{year}-{month}-01", end=f"{year + 1}-01-1", freq="D"
                )[:-1]
            return pd.date_range(
                start=f"{year}-{month}-01", end=f"{year}-{month + 1}-1", freq="D"
            )[:-1]

        first_of_month = pd.Timestamp(f"{self.year}-{self.month}-01").day_of_week

        mo = self.this_month_df.copy()

        mo.loc[:, "date"] = mo["date"].dt.date

        # Fill in missing days
        mo = pd.DataFrame(all_days(self.month, self.year), columns=["date"]).merge(
            mo, on="date", how="left"
        )

        mo.loc[:, "dow"] = mo["date"].dt.day_of_week
        mo.loc[:, "week"] = (mo["date"].dt.day - 1 + first_of_month) // 7 + 1

        calendar_sum = (
            mo.groupby(["week", "dow"])
            .agg({"date": "first", "body": "count"})
            .rename(columns={"body": "count"})
            .reset_index()
        )

        # Generate plot
        plot = (
            alt.Chart(calendar_sum)
            .mark_point(
                shape="square",
                size=2000,
                filled=True,
                opacity=1,
            )
            .encode(
                x=alt.X(
                    "dow:O",
                    axis=alt.Axis(
                        orient="top",
                        domain=False,
                        ticks=False,
                        labelExpr="{'0': 'M', '1': 'T', '2': 'W', '3': 'Th', '4': 'F', '5': 'S', '6': 'Su'}[datum.value]",
                        title=None,
                        labelAngle=0,
                        labelPadding=10,
                    ),
                ),
                y=alt.Y("week:O", axis=None),
                color=alt.Color("count:Q", scale=alt.Scale(scheme="lightorange")),
                tooltip=["date", "count"],
            )
            .properties(
                width=500,
                height=300,
            )
            .configure_view(stroke=None)
            .interactive()
        )

        return plot

    ### Simple Stats ###

    def simple_stats(
        self, likelihood_min=10, likelihood_buffer=10, convo_time=12
    ) -> dict:
        output = {}
        mo = self.this_month_df.copy()
        dates = mo["date"].dt.date

        # Texts per day
        output["avg"] = mo.shape[0] / Recap.days_in_month(self.month, self.year)

        # Median texts per day
        output["median"] = mo.groupby(dates).size().median()

        # Anastasiya Ratio
        ratio = mo.groupby("is_from_me").size()
        output["ratio"] = ratio[0] / ratio[1]

        # Number of i love you
        output["ilys"] = mo["body"].str.lower().str.count("i love you").sum()

        # Max Day
        max_day = mo.groupby(dates).size().sort_values(ascending=False).head(1)
        output["max_day"] = (max_day.index[0], max_day.values[0])

        # Favorite Hour
        hours = mo["date"].dt.hour
        output["fav_hour"] = hours.value_counts().idxmax()

        # Longest Conversation
        def longest_convo() -> pd.Timedelta:
            time_diff = mo["date"].diff()
            same_convo = time_diff < pd.Timedelta(minutes=convo_time)
            falses = pd.DataFrame(same_convo[same_convo == False])
            falses["convo"] = range(falses.shape[0])
            same_convo = pd.DataFrame(same_convo)
            same_convo = same_convo.merge(
                falses["convo"], left_index=True, right_index=True, how="left"
            )
            same_convo = same_convo.ffill()

            return (
                mo.groupby(same_convo["convo"])["date"]
                .agg(lambda x: x.max() - x.min())
                .sort_values(ascending=False)
                .iloc[0]
            )

        output["longest_convo"] = longest_convo()

        # Combined Word Count
        total = Recap.combined_count(mo, on="word")
        total = total[(total["count_me"] + total["count_you"]) > likelihood_min]

        # Calculate log likelihood with a given buffer
        log = np.log(
            ((total["count_me"] + likelihood_buffer) / total["count_me"].sum())
            / ((total["count_you"] + likelihood_buffer) / total["count_you"].sum())
        )

        output["chris"] = log.idxmax()
        output["anastasiya"] = log.idxmin()

        return output

    ### Word Rankings ###

    def word_rankings(
        self,
        plot_width: int = 400,
        plot_height: int = 600,
        n: int = 10,
    ) -> alt.Chart:

        mo = self.this_month_df.copy()
        last_mo = self.last_month_df.copy()

        # calculate word counts
        month_ranks = Recap.df_word_occurrences(mo, ranks=True)
        last_ranks = Recap.df_word_occurrences(last_mo, ranks=True)

        month_ranks["movement"] = last_ranks["rank"] - month_ranks["rank"]

        me = mo[mo["is_from_me"] == 1]
        you = mo[mo["is_from_me"] == 0]

        me_counts = Recap.df_word_occurrences(me, ranks=False)
        you_counts = Recap.df_word_occurrences(you, ranks=False)

        # Add counts to the ranks
        month_ranks["me"] = me_counts["count"]
        month_ranks["you"] = you_counts["count"]

        # Assign colors for annotations
        month_ranks["move_color"] = month_ranks["movement"].apply(
            lambda x: 1 if x > 0 else -1 if x < 0 else 0
        )

        # Convert numbers to strings for annotations
        month_ranks["move_str"] = month_ranks["movement"].fillna("NEW")
        month_ranks["move_str"] = month_ranks["move_str"].apply(
            lambda x: (
                x
                if x == "NEW"
                else (
                    "-"
                    if x == 0
                    else f"⏶ {str(int(x))}" if x > 0 else f"⏷ {str(int(x) * -1)}"
                )
            )
        )

        month_ranks[["me", "you"]] = month_ranks[["me", "you"]].fillna(0)

        # Take top N words
        top = month_ranks.sort_values("rank").head(n).reset_index()

        # Make Plot
        final = Recap.ranking_plot(top, plot_width, plot_height)

        return final

    @staticmethod
    def ranking_plot(
        top: pd.DataFrame, plot_width: float, plot_height: float
    ) -> alt.Chart:
        config: dict[float, str] = {
            "overall_width": plot_width,
            "overall_height": plot_height,
            "bar_size": 0.35,
            "color": "#E07F67",
            "bar_height": 50,
        }
        config["middle_size"] = 1 - 2 * config["bar_size"]

        you, me = Recap.ranking_bars(top, config)
        middle = Recap.ranking_text(top, config)

        final: alt.Chart = alt.hconcat(you, middle, me, spacing=0)
        final = final.configure_view(stroke=None)

        return final

    @staticmethod
    def ranking_bars(top: pd.DataFrame, config: dict) -> tuple[alt.Chart, alt.Chart]:
        domain_max: float = max(top["me"].max(), top["you"].max())

        you = Recap.ranking_bar(top, config, domain_max, left=True)
        me = Recap.ranking_bar(top, config, domain_max, left=False)

        return you, me

    @staticmethod
    def ranking_bar(
        top: pd.DataFrame, config: dict, domain_max: float, left=True
    ) -> alt.Chart:
        # Unpack config
        overall_width = config["overall_width"]
        overall_height = config["overall_height"]
        bar_size = config["bar_size"]
        color = config["color"]

        # Left or right
        title = "Anastasiya" if left else "Chris"
        text_align = "right" if left else "left"
        column = "you" if left else "me"
        reverse = left

        # Create the bar chart
        bars = (
            alt.Chart(top)
            .mark_bar(
                color=color,
                # height=bar_height,
            )
            .encode(
                x=alt.X(
                    f"{column}:Q",
                    axis=alt.Axis(
                        title=title,
                        orient="top",
                        grid=False,
                        domain=False,
                        ticks=False,
                        labels=False,
                    ),
                    scale=alt.Scale(domain=[0, domain_max], reverse=reverse),
                ),
                y=alt.Y("order_rank:O", axis=None),
                # tooltip=['rank', 'count', 'movement', 'me', 'you']
            )
            .properties(
                width=overall_width * bar_size,
                height=overall_height,
            )
        )

        text = bars.mark_text(
            align=text_align,
            baseline="middle",
            dx=5 * (-1) ** reverse,  # Adjust the position of the text
        ).encode(text=f"{column}:Q")

        plot = bars + text

        return plot

    @staticmethod
    def ranking_text(top: pd.DataFrame, config: dict) -> alt.Chart:
        overall_width: float = config["overall_width"]
        overall_height: float = config["overall_height"]
        middle_size: float = config["middle_size"]

        middle_text = (
            alt.Chart(top)
            .mark_text(align="center", baseline="middle", dx=0, dy=0, fontSize=15)
            .encode(
                text="word:N",
                y=alt.Y("order_rank:O", axis=None),
            )
            .properties(
                width=overall_width * middle_size,
                height=overall_height,
            )
        )

        rank_text = (
            alt.Chart(top)
            .mark_text(
                align="left",
                baseline="middle",
                dx=-40,  # Adjust the position of the rank text
                fontSize=12,
            )
            .encode(
                text="rank:O",
                y=alt.Y("order_rank:O", axis=None),
            )
        )

        movement_text = (
            alt.Chart(top)
            .mark_text(align="center", baseline="middle", dx=40, dy=0, fontSize=12)
            .encode(
                text="move_str:O",
                y=alt.Y("order_rank:O", axis=None),
                color=alt.Color(
                    "move_color:N",
                    scale=alt.Scale(domain=[-1, 0, 1], range=["red", "black", "green"]),
                    legend=None,
                ),
            )
        )

        middle: alt.Chart = rank_text + middle_text + movement_text

        return middle

    ### Comparative Line Chart ###

    def month_line(self):
        end_date = pd.Timestamp(f"{self.year}-{self.month + 1}-1") - pd.DateOffset(
            days=1
        )

        # Only include the last 12 months
        this_year = self.df[
            (self.df["date"] > end_date - pd.DateOffset(months=12))
            & (self.df["date"] <= end_date)
        ]
        this_year["month"] = self.df["date"].dt.month
        this_year["month_str"] = self.df["date"].dt.month_name()
        this_year["day"] = self.df["date"].dt.day

        # Build the cumulative sum
        cumulative = (
            this_year.groupby(["month", "day"]).size().reset_index(name="count")
        )
        cumulative["cumulative"] = cumulative.groupby("month")["count"].cumsum()

        # Get the last day of each month for text annotation
        last_days = cumulative.groupby("month").tail(1)[["month", "day"]]
        cumulative = cumulative.merge(last_days, on="month", suffixes=("", "_last"))

        # Get the month name for text annotation
        cumulative["month_str"] = cumulative["month"].apply(
            lambda x: pd.Timestamp(f"{self.year}-{x}-1").month_name()
        )

        # Build plot
        final = self.cumulative_line(cumulative)

        return final

    def cumulative_line(self, cumulative: pd.DataFrame) -> alt.Chart:
        line_plot = (
            alt.Chart(cumulative)
            .mark_line(interpolate="monotone")
            .encode(
                x=alt.X("day:O", axis=alt.Axis(title="Day of the Month")),
                y=alt.Y("cumulative:Q", axis=alt.Axis(title="Number of Messages")),
                color=alt.condition(
                    alt.datum.month == self.month,
                    alt.value("#E07F67"),
                    alt.value("gray"),
                ),
                size=alt.condition(
                    (alt.datum.month == self.month)
                    | (alt.datum.month == self.last_month),
                    alt.value(4),
                    alt.value(2),
                ),
                opacity=alt.condition(
                    (alt.datum.month == self.month)
                    | (alt.datum.month == self.last_month),
                    alt.value(1),
                    alt.value(0.6),
                ),
                detail="month:N",
            )
            .properties(width=800, height=200)
        )

        text_annotation = (
            alt.Chart(cumulative)
            .mark_text(align="left", baseline="middle", dx=5)
            .encode(
                x=alt.value(800), y=alt.Y("cumulative:Q"), text="month_cumulative:N"
            )
            .transform_calculate(
                month_cumulative='datum.month_str + " - " +format(datum.cumulative, ",")'
            )
            .transform_filter(
                (alt.datum.month == self.month) | (alt.datum.month == self.last_month)
            )
            .transform_filter(
                alt.datum.day
                == alt.datum.day_last  # Adjust this to the last day of the month or the day you want to annotate
            )
        )

        final = line_plot + text_annotation

        final = final.configure_view(stroke=None)

        return final

    ### Percent over prior ###

    def percent_prior_month(self, percentage=True):
        try:
            return (
                self.this_month_df.shape[0]
                / self.last_month_df.shape[0]
                * (100**percentage)
            )  # Returns as a percentage or proportion
        except ZeroDivisionError:
            return 0

    ### Breakout/Forgotten ###
    def breakout_forgotten(self) -> dict:
        month_words = Recap.df_word_occurrences(self.this_month_df)
        last_month_words = Recap.df_word_occurrences(self.last_month_df)
        prior_months_words = Recap.df_word_occurrences(self.prior_months_df)

        month_emojis = Recap.count_emojis(self.this_month_df)
        last_month_emojis = Recap.count_emojis(self.last_month_df)
        prior_months_emojis = Recap.count_emojis(self.prior_months_df)

        output = {
            "breakout_word": Recap.breakout_word(month_words, prior_months_words),
            "breakout_emoji": Recap.breakout_emoji(month_emojis, prior_months_emojis),
            "forgotten_word": Recap.forgotten_word(month_words, last_month_words),
            "forgotten_emoji": Recap.forgotten_emoji(month_emojis, last_month_emojis),
        }

        return output

    @staticmethod
    def breakout_word(
        month_words: pd.DataFrame,
        prior_words: pd.DataFrame,
        min_uses: int = 10,
        buffer: int = 15,
    ) -> dict:
        total = pd.merge(
            month_words, prior_words, on="word", suffixes=("_mo", "_no"), how="outer"
        ).fillna(0)
        total = total[(total["count_mo"] + total["count_no"]) >= min_uses]
        total["likelihood"] = (
            (total["count_mo"] + buffer) / total["count_mo"].sum()
        ) / ((total["count_no"] + buffer) / total["count_no"].sum())
        total["true_likelihood"] = (total["count_mo"] / total["count_mo"].sum()) / (
            (total["count_no"] + 1e-6) / total["count_no"].sum()
        )

        # Best word
        best_word = (
            total.sort_values("likelihood", ascending=False)
            .head(1)
            .reset_index()
            .to_dict(orient="records")[0]
        )

        return best_word

    @staticmethod
    def breakout_emoji(
        month_emojis: pd.DataFrame, prior_emojis: pd.DataFrame, min_uses=5, buffer=5
    ) -> dict:
        total = pd.merge(
            month_emojis, prior_emojis, on="emoji", suffixes=("_mo", "_no"), how="outer"
        ).fillna(0)
        total = total[(total["count_mo"] + total["count_no"]) >= min_uses]
        total["likelihood"] = (
            (total["count_mo"] + buffer) / total["count_mo"].sum()
        ) / ((total["count_no"] + buffer) / total["count_no"].sum())
        total["true_likelihood"] = (total["count_mo"] / total["count_mo"].sum()) / (
            (total["count_no"] + 1e-6) / total["count_no"].sum()
        )

        # Best emoji
        best_emoji = (
            total.sort_values("likelihood", ascending=False)
            .head(1)
            .reset_index()
            .to_dict(orient="records")[0]
        )

        return best_emoji

    @staticmethod
    def forgotten_word(month_words: pd.DataFrame, last_words: pd.DataFrame):
        total = pd.merge(
            month_words, last_words, on="word", suffixes=("_mo", "_last"), how="outer"
        ).fillna(0)
        total["num_fewer"] = total["count_last"] - total["count_mo"]

        return (
            total.sort_values("num_fewer", ascending=False)
            .head(1)
            .reset_index()
            .to_dict(orient="records")[0]
        )

    @staticmethod
    def forgotten_emoji(month_emojis: pd.DataFrame, last_emojis: pd.DataFrame):
        total = pd.merge(
            month_emojis,
            last_emojis,
            on="emoji",
            suffixes=("_mo", "_last"),
            how="outer",
        ).fillna(0)
        total["num_fewer"] = total["count_last"] - total["count_mo"]

        return (
            total.sort_values("num_fewer", ascending=False)
            .head(1)
            .reset_index()
            .to_dict(orient="records")[0]
        )
