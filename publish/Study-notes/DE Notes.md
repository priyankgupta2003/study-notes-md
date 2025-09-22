
- Shuffle
    
    - If we are using teamwork, we need to guarantee certain data is on a certain machine (like if we are counting how many messages each user has received). The team accomplishes this guarantee by passing all of your data to one machine via shuffle (example in the diagram below). We only HAVE to do this when we do **GROUP BY, JOIN,** or **ORDER BY. (**[a quick 2 minute video about how I managed this at petabyte scale at Netflix](https://www.youtube.com/watch?v=g23GHqJje40))
        
        [
        
        ![Spark Shuffle Design](https://substackcdn.com/image/fetch/$s_!BKRM!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fad094a38-8711-446c-8461-998988c1b901_317x289.png "Spark Shuffle Design")
        
        
        
        ]
        
    - Shuffling isn’t a bad thing remember! It actually is really good because it makes distributed compute mostly the same as single node compute! The only time it gets in the way of things is at very large scale!
        
    - Things you should consider to reduce shuffling at very large scale
        
        - Broadcast **JOIN**
            
            - If one side of your **JOIN is small (< 5 GBs), you can “broadcast”** the entire data set to your executors. This allows you to do the join without shuffle which is much faster
                
        - Bucket **JOIN**
            
            - If both sides of your **JOIN are large, you can bucket them first and then do the join**, this allows. Remember you’ll still have to shuffle the data once to bucket it, but if you’re doing multiple **JOIN** with this data set it will be worth it!
                
        - Partitioning your data set
            
            - Sometimes you’re just trying to **JOIN** too much data because you should **JOIN** one day of data not multiple. Think about how you could do your **JOIN** with less data
                
        - [Leverage cumulative table design](https://github.com/EcZachly/cumulative-table-design)
            
            - Sometimes you’ll be asked to aggregate multiple days of data for things like “monthly active users.” Instead of scanning thirty days of data, leverage cumulative table design to dramatically improve your pipeline’s performance!
                
    - Shuffle can have problems too! What if one team member gets a lot more data than the rest? This is called skew and happens rather frequently! There are a few options here:
        
        - In Spark 3+, you can enable adaptive execution. This solves the problem very quickly and I love Databricks for adding this feature!
            
        - In Spark <3, you can [salt the](https://medium.com/curious-data-catalog/sparks-salting-a-step-towards-mitigating-skew-problem-5b2e66791620) **[JOIN](https://medium.com/curious-data-catalog/sparks-salting-a-step-towards-mitigating-skew-problem-5b2e66791620)** [or](https://medium.com/curious-data-catalog/sparks-salting-a-step-towards-mitigating-skew-problem-5b2e66791620) **[GROUP BY](https://medium.com/curious-data-catalog/sparks-salting-a-step-towards-mitigating-skew-problem-5b2e66791620).** Salting allows you to leverage random numbers so you get a more even distribution of your workload among your team members!