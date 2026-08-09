# הוספת חשבון למאגר

כל חשבון הוא תיקייה אחת תחת `~/.kaggle-accounts/`, בשם כרצונך — השם הוא מה
ש־`queue.py` מקבל כארגומנט `account`:

```text
~/.kaggle-accounts/
  yossi/
    access_token
  miriam/
    credentials.json
```

## הדרך הרגילה: טוקן

ב־Kaggle: **Settings ← API ← Generate New Token**. מה שמתקבל הוא מחרוזת אחת. שומרים
אותה כקובץ טקסט פשוט בשם `access_token` — הקובץ מכיל את הטוקן ותו לא, בלי JSON ובלי
מרכאות:

```bash
mkdir -p ~/.kaggle-accounts/yossi
printf '%s' 'THE_TOKEN' > ~/.kaggle-accounts/yossi/access_token
chmod 600 ~/.kaggle-accounts/yossi/access_token
```

שם המשתמש אינו נדרש — הטוקן מזהה את החשבון בעצמו.

> גרסאות ישנות של ה־CLI השתמשו ב־`kaggle.json` עם `username` ו־`key`. גרסה 2.2 כבר לא
> קוראת אותו: היא מחפשת `KAGGLE_API_TOKEN` או `~/.kaggle/access_token`. אם קיבלת
> `kaggle.json`, קח ממנו את הערך של `key` ושים אותו ב־`access_token`.

## הדרך השנייה: OAuth, בלי למסור טוקן

מי שמעדיף לא למסור מפתח כלל יכול להזדהות בעצמו מול הדפדפן:

```bash
KAGGLE_CONFIG_DIR=~/.kaggle-accounts/miriam kaggle auth login
```

זה משאיר `credentials.json` באותה תיקייה, ו־`queue.py` עובד איתו זהה.

## אימות

```bash
python3 tools/kaggle/queue.py quota
```

מדפיס שורה לכל חשבון שיש לו הזדהות, עם המכסה שבאמת נותרה בו. חשבון שאינו מופיע או
מדווח על כישלון — הטוקן שגוי או שהקובץ ריק.

## למה מחוץ למאגר

טוקן הוא אמצעי הזדהות: מי שמחזיק בו יכול לפעול כאותו חשבון. לכן הוא לא צריך להיות
במרחק `git add -A` מפרסום. כל אחד מחברי הצוות יכול לבטל את הטוקן שלו מאותו מסך הגדרות,
וזו הדרך הנקייה לסיים השתתפות.

המכסה היא לכל חשבון בנפרד ואינה מתאחדת — 30 שעות GPU לשבוע לכל חשבון.
