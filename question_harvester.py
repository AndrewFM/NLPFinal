import stackexchange
import time

#Sites we're interested in answering questions about
site_list = [stackexchange.StackOverflow,
			 stackexchange.Mathematics,
			 stackexchange.Photography,
			 stackexchange.Arqade,
			 stackexchange.GameDevelopment,
			 stackexchange.WebApplications,
			 stackexchange.SeasonedAdvice,
			 stackexchange.HomeImprovement,
			 stackexchange.PersonalFinanceampMoney,
			 stackexchange.EnglishLanguageampUsage,
			 stackexchange.Physics,
			 stackexchange.Writers,
			 stackexchange.GraphicDesign,
			 stackexchange.VideoProduction,
			 stackexchange.ScienceFictionampFantasy,
			 stackexchange.PhysicalFitness,
			 stackexchange.Parenting,
			 stackexchange.MusicPracticeampTheory,
			 stackexchange.Philosophy,
			 stackexchange.GardeningampLandscaping,
			 stackexchange.Travel,
			 stackexchange.MoviesampTV,
			 stackexchange.Biology,
			 stackexchange.Sports,
			 stackexchange.Academia,
			 stackexchange.Chemistry,
			 stackexchange.Robotics,
			 stackexchange.Politics,
			 stackexchange.AnimeampManga,
			 stackexchange.Pets,
			 stackexchange.Astronomy,
			 stackexchange.EarthScience,
			 stackexchange.SoundDesign,
			 stackexchange.TheWorkplace,
			 stackexchange.TheGreatOutdoors,
			 stackexchange.History]

#Extract questions from each of those sites, and store them later for our future purposes.
#This will be the training data for our site classification task.

num_questions = 2500 #Number of questions per site to extract
user_api_key = "I9AkihbZDuFtUu)8rxEi*A(("

for s in site_list:
	print("Collecting questions from "+s)
	outfile = open(s+'.txt', 'wb')
	outfile.close()
	site = stackexchange.Site(s, impose_throttling = True, app_key=user_api_key)

	q_so_far = 0
	questions = site.recent_questions(pagesize=100)

	while q_so_far < num_questions:
		for q in questions:
			outfile = open(s+'.txt', 'ab')
			outfile.write((q.title+'\n').encode('utf-8'))
			outfile.close()
			q_so_far += 1

			if q_so_far > num_questions:
				break
		time.sleep(1)

		if q_so_far < num_questions:
			questions = questions.extend_next()

	time.sleep(1) #Wait one second, don't want to go over the API throttle.
