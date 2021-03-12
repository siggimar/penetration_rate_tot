# extract total only soundings from SGF file
def get_tot_soundings_from_sgf( raw_sgf ):
    all_soundings = raw_sgf.split( '$\n' )
    tot_soundings = []
    
    for sounding in all_soundings:
        if len(sounding)>len(sounding.upper().replace("HM=24", "")):
            tot_soundings.append('$\n' + sounding)
    return tot_soundings

# read file to string
def read_file( file_path ):
    file_contents = ''

    try:
        with open( file_path, 'r') as open_file: # open-read-close
            file_contents = open_file.read()
    except Exception as e:
        print( e )
    return file_contents

# wrapper to read file and get tot contents
def read_tot( file_path, sounding_nr = 0 ):
    raw_contents = read_file( file_path )
    tot_soundings = get_tot_soundings_from_sgf( raw_contents )[sounding_nr]
    return tot_soundings.strip()