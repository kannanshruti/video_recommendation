import vrec
import argparse

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', dest='query', help='Clip ID for which recommendations required',
                        default=214566929, type=int)
    parser.add_argument('-n', dest='n', help='No of recommendations required',
                        default=10, type=int)
    args = parser.parse_args()
    query = args.query
    n = args.n

    # Specify data folder paths
    folder = '/home/shruti/Documents/code_sk/Datasets/similar-staff-picks-challenge/'
    categories = 'similar-staff-picks-challenge-categories.csv'
    clip_categories = 'similar-staff-picks-challenge-clip-categories.csv'
    clips = 'similar-staff-picks-challenge-clips.csv'

    # Creating object and analysing data
    vr = vrec.VideoRecommendation(folder, categories, clip_categories, clips)
    print 'DataFrame Shape:'
    print ' categories:', vr.categories.shape
    print ' clip_categories:', vr.clip_categories.shape
    print ' clips:', vr.clips.shape
    print ' combined data:', vr.combined_data.shape, '\n'

    # Extract part of dataset with column where we don't want NaNs
    dataset1 = vr.get_dataset('title')
    print '#Clips in total:', vr.combined_data['id'].count()
    print '#Clips without titles:', len(vr.combined_data) - vr.combined_data['title'].count()
    print '#Clips extra in clip_categories', len(vr.combined_data) - len(vr.clips), '\n'

    # Extract words from the required columns
    sentences = {}
    sentences = vr.get_sentences(dataset1, 'title', include_caption=False)

    # Create the model for the extracted words
    model = vr.get_model(sentences)

    # Get the vector for each word from the model
    sen_vector = {}  # Vector for each word per clipID
    avg_vector = {}  # Average vector for each clipID
    sen_vector, avg_vector = vr.get_vector(model, sentences)

    # Get a JSON object for each clipID
    json_dict = {}
    json_dict = vr.get_json(dataset1)

    # Get closest matches for query clipID
    res = vr.get_closest_match(query, avg_vector, 10)
    query_json = vr.get_json_list(json_dict, [res])
    print query_json
    # vr.disp_json(query_json)  # Uncomment for a more readable output

    # Generate JSON objects for the specified list of clipIDs
    matches = []
    closest_match = []
    queries = [14434107, 249393804, 71964690, 78106175, 228236677,
               11374425, 93951774, 35616659, 112360862, 116368488]
    for query in queries:
        matches = vr.get_closest_match(query, avg_vector, 10)
        closest_match.append(matches)

    # Generate the final JSON list of JSON objects
    json_list = vr.get_json_list(json_dict, closest_match,
                                 output_file='results.json')
