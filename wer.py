import jiwer
ref = [
    "That's definitely the moment from this year that I'll take away and remember from 2016.",
    "Here, it's very different. Monaco is very open. We get to see a lot more of the fans because as we walk to the pit lane, there are fans everywhere up on the hills and the grandstands.",
    "And yeah, it would be nice to have a bit more of that. I mean, we do have fans signing all around the world now, so it has got better over the years, but yeah, it's...",
    "Fans would like to see more of Formula One as a whole from inside the paddock and not just through your eyes because you're very good, but sometimes you're incorrect.",    
    "to find out about. But we try to get up to the, well, personally try",
    "So that's got to be my aim. But I think this year is probably more tricky than any",
    "But in the high-speed corners, you can feel the car moving around a bit more. But a great car, and I really, really enjoyed it. I'm just disappointed.",
    "For me it's been a great race, racing in Albert Park. The last two years I've won here, so I come here."
]

# hyp = [
#     "and that definitely helped fuel the love to get away. I remember from 2016,",
#     "Yeah, it's very different from all of our other work. We're a little more of a fan of the genre.",
#     "And to help me learn to skate would be perfect. I mean, we do a lot of skating throughout the world there. So I had to go back to the rink.",
#     "I mean, it's more of a weather song for me. It's kind of about the elements and the great experience of going on the road. Sometimes it's very relaxing.",
#     "fun. I mean, we're just working on the game. We're just preparing to the game.",
#     "So it's tough to pick up, but I think it's probably more tricky than anything.",
#     "but yeah, it's been good. I'm loving every minute of it, but I'm a great guy, I really enjoy it.",
#     "For me, it's been a great ride. I've had the time of my life. I think I've really got to come here."
# ]


hyp = [
    "That's definitely the moment for us. I'm taking away President Obama.",
    "They're fighting with spines. They hold their spines. It was like we shot a couple of shots. It was like, we shot a couple of shots. The hot skis just go down the hill. It's quite surreal.",
    "And yeah, it would be nice to have a bit more of that. I mean, we do have fans signing all around the world now, so it has got better over the years, but yeah, it's...",
    "I'd like to see more of the first one as a whole. From inside the paddock, not just through your eyes, but it's your very good. It's a very good story.",
    "When we initially got the offer, but we tried to get it to the, well, push it where we like it.",
    "and shows that's gonna be like. You know, it's really probably more like.",
    "but yeah, it's been good. I'm loving every minute of it, but I'm a great guy, I really enjoy it.",
    "For me, it's been a great ride. I've had the time of my life. I think I've really got to come here."
]

out = jiwer.process_words(
    reference=ref,
    hypothesis=hyp,
    reference_transform=jiwer.wer_standardize, 
    hypothesis_transform=jiwer.wer_standardize
)

print(f"WER: {out.wer:.4f}")
