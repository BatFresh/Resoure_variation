from jtop import jtop, JtopException
import csv
import argparse
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple jtop logger')
    # Standard file to store the logs
    parser.add_argument('--file', action="store", dest="file", default="log.csv")
    args = parser.parse_args()

    print("Simple jtop logger")
    print("Saving log on {file}".format(file=args.file))

    try:
        with jtop() as jetson:
            # Make csv file and setup csv
            with open(args.file, 'w') as csvfile:
                stats = jetson.stats

                stats["GPU Memory"] = 0
                stats["used Memory"] = 0
                stats["total Memory"] = 0

                # Initialize cws writer

                writer = csv.DictWriter(csvfile, fieldnames=stats.keys())
                # Write header
                writer.writeheader()
                # Write first row
                writer.writerow(stats)
                # Start loop
                while True:
                    stats = jetson.stats
                    stats["GPU Memory"] = jetson.ram["shared"]/1024
                    stats["used Memory"] = jetson.ram["use"]/1024
                    stats["total Memory"] = jetson.ram["tot"]/1024
                    # Write row
                    writer.writerow(stats)
                    print("Log at {time}".format(time=stats['time']))
                    time.sleep(0.1)
                print("log finsihed")
    except JtopException as e:
        print(e)
    except KeyboardInterrupt:
        print("Closed with CTRL-C")
    except IOError:
        print("I/O error")

# EOF
