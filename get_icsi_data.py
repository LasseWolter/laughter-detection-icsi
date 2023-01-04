from lhotse.recipes.icsi import download_icsi, prepare_icsi
def main() -> None:
    try: 
        download_icsi(audio_dir='./data/icsi', transcripts_dir='./data/icsi')
    except Exception as err:
        print("Something went wrong while downloading ICSI files.", err)

if __name__ == '__main__': 
    main()
