"""
Psychopy middleware. Provides display / stimulus functionality for the experiment.
"""

from psychopy import visual, event, monitors, core, data

class Psypy:
    """
    Wrapper class for reusable psychopy activities

    Attributes
    ----------
    conf : dict
        Settings for psychopy. See config.psy
    mon : monitors.Monitor
        The psychopy active monitor configuration
    win : visual.Window
        The psychopy active window to display stimuli
    wait_keys : list
        Keys that will be waited for on psychopy stimuli
    """

    conf: dict
    mon: monitors.Monitor
    win: visual.Window
    wait_keys: list = ['space', 'escape']

    def __init__(self, conf: dict) -> None:
        self.conf = conf
        self.mon = self.prepare_monitor()
        self.win = self.get_window()

    def __del__(self) -> None:
        """
        Close the psychopy window and shut down the experiment
        """
        self.win.close()

    def prepare_monitor(self) -> monitors.Monitor:
        """
        Set up psychopy monitor
        """

        mon = monitors.Monitor(self.conf.get('monitorName'))
        mon.setSizePix(self.conf.get('monitorSize'))
        mon.setWidth(self.conf.get('monitorWidth'))
        return mon

    def wait_for_key(self) -> list:
        """
        Just a reusable wrapper for the psychopy key event
        """

        # Only listen for keys we want
        key = event.waitKeys(keyList=self.wait_keys)
        # If key is esc, leave the experiment
        if 'escape' in key:
            core.quit()
        return key
    
    def wait_for_time(self, time: float) -> None:
        """
        Wait for a given amount of time
        """
        core.wait(time)

    def get_window(self) -> visual.Window:
        """
        Return a window for drawing stimuli
        """

        return visual.Window(
            size=self.conf.get('windowSize'),
            fullscr=self.conf.get('fullScreen'),
            monitor=self.mon,
            color=self.conf.get('windowColor'))

    def display_text_message(self, txt: str, wait: bool = True, wait_time: int = None, units="cm", height=0.5) -> None:
        """
        Display psychopy message / instructions
        """

        msg = visual.TextStim(self.win, text=txt.strip(), color=self.conf.get('textColor'), units=units, height=height)
        msg.draw()
        self.win.flip()
        if wait:
            self.wait_for_key()
        if wait_time:
            self.wait_for_time(wait_time)

    def hide_cursor(self) -> None:
        """
        Hide the cursor
        """
        self.win.setMouseVisible(False)
    
    def show_cursor(self) -> None:
        """
        Show the cursor
        """
        self.win.setMouseVisible(True)

    def display_text_sequence(self, txt: str) -> list:
        """
        Display word by word text sequence
        """

        # Prepare timer
        stopwatch = core.Clock()
        sequence_data = []
        sequence = 1
        # Display the given text word by word
        for word in txt.split():
            # ignore blank / non-words
            word = word.strip()
            if word == '':
                continue
            # Prepare the word display
            msg = visual.TextStim(self.win, text=word, color=self.conf.get('textColor'))
            msg.draw()
            # Show the word
            self.win.flip()
            # Start timer
            stopwatch.reset()
            self.wait_for_key()
            time = stopwatch.getTime()
            # Collate data for this word
            sequence_data.append({
                'timestamp': data.getDateStr(format='%Y-%m-%d %H:%M:%S'),
                'word': word,
                'time': time,
                'sequence': sequence
            })
            sequence += 1
        return sequence_data

    def display_image(self, img: str, wait: int = 5) -> None:
        """
        Display psychopy image
        """
        img = visual.ImageStim(self.win, image=img)
        img.draw()
        self.win.flip()
        if wait:
            self.wait_for_time(wait)
    
    def wait_with_callback(self, duration: float, onStart: callable = None, onEnd: callable = None) -> None:
        """
        Call a callback function, wait for a keypress and call the second callback function
        """
        if onStart:
            onStart()
        self.wait_for_time(duration)
        if onEnd:
            onEnd()

    def call_on_next_flip(self, callback: callable, *args: any) -> None:
        """
        Call a callback function on the next flip
        """
        self.win.callOnFlip(callback, *args)

def shutdown_psychopy() -> None:
    """
    Shutdown psychopy
    """
    core.quit()